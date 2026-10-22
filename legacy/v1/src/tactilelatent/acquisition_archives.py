"""Seekable archive readers; never materialize a second copy of TacQuad."""
from collections import OrderedDict
import bisect
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import time
import zipfile

import numpy as np
from PIL import Image

from .aligned_frames import file_hash, write_json


def safe_member(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe archive member')
    return p


class MultipartFile(io.RawIOBase):
    """Concatenated split ZIP, including reads crossing part boundaries."""
    def __init__(self, paths):
        super().__init__()
        self.paths=[Path(p) for p in paths]
        self.starts=[0]
        for p in self.paths:self.starts.append(self.starts[-1]+p.stat().st_size)
        self.position=0; self.handles={}

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self, offset, whence=0):
        value=offset if whence==0 else self.position+offset if whence==1 else self.starts[-1]+offset if whence==2 else -1
        if value<0:raise ValueError('Invalid seek')
        self.position=value;return value

    def read(self, size=-1):
        remaining=self.starts[-1]-self.position
        if size<0:size=remaining
        size=min(size,remaining)
        result=[]
        while size>0:
            part=bisect.bisect_right(self.starts,self.position)-1
            handle=self.handles.get(part)
            if handle is None:self.handles[part]=handle=self.paths[part].open('rb')
            handle.seek(self.position-self.starts[part])
            data=handle.read(min(size,self.starts[part+1]-self.position))
            if not data:raise EOFError('Split archive ended early')
            result.append(data);size-=len(data);self.position+=len(data)
        return b''.join(result)

    def readinto(self, buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)

    def close(self):
        for handle in self.handles.values():handle.close()
        super().close()


class RemoteRangeFile(io.RawIOBase):
    """Bounded HTTP Range reader, used to select T-Dex episodes inside a ZIP.

    A missing/incorrect Content-Range fails rather than downloading 31 GB into
    memory. The pinned central-directory digest detects changed Drive content.
    """
    def __init__(self, url, size, params=None, block_size=2**20):
        import httpx
        super().__init__()
        self.client=httpx.Client(follow_redirects=True,timeout=120)
        self.url=url;self.size=size;self.params=params or {};self.block_size=block_size
        self.position=0;self.cache=OrderedDict();self.bytes_received=0

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self, offset, whence=0):
        value=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if value<0:raise ValueError('Invalid remote seek')
        self.position=value;return value

    def _block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index);return self.cache[index]
        lo=index*self.block_size;hi=min(lo+self.block_size,self.size)-1
        params=dict(self.params,range_block=f'{lo}-{hi}')
        last=None
        for attempt in range(4):
            try:
                with self.client.stream('GET',self.url,params=params,
                    headers={'Range':f'bytes={lo}-{hi}','Accept-Encoding':'identity'}) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{self.size}':
                        raise ValueError('Server did not honor bounded byte range')
                    parts=[];received=0
                    for chunk in response.iter_bytes(2**20):
                        received+=len(chunk)
                        if received>hi-lo+1:raise ValueError('Range response overflow')
                        parts.append(chunk)
                    data=b''.join(parts)
                if len(data)!=hi-lo+1:raise ValueError('Incomplete range')
                break
            except Exception as error:
                last=error;time.sleep(2**attempt)
        else:raise last
        self.bytes_received+=len(data);self.cache[index]=data
        while len(self.cache)>8:self.cache.popitem(last=False)
        return data

    def read(self,size=-1):
        if size<0:size=self.size-self.position
        size=min(size,self.size-self.position);result=[]
        while size>0:
            index,offset=divmod(self.position,self.block_size)
            data=self._block(index)[offset:offset+size]
            if not data:raise EOFError('Remote archive ended early')
            result.append(data);self.position+=len(data);size-=len(data)
        return b''.join(result)

    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)

    def close(self):
        self.client.close();super().close()


def tacquad(job):
    parts=[]
    for item in job.inventory()['files']:
        p=item['path']
        if p=='README.md' or p.startswith('tacquad.zip.'):
            path=job.download(p)
            if p.startswith('tacquad.zip.'):parts.append(path)
    if len(parts)!=7:raise ValueError('TacQuad requires all seven core archive parts')
    parts.sort();dbpath=job.root/'members.sqlite'
    temp=dbpath.with_suffix('.partial.sqlite')
    if temp.exists():temp.unlink()
    database=sqlite3.connect(temp)
    database.execute('CREATE TABLE members(name TEXT PRIMARY KEY, offset INTEGER, bytes INTEGER, compressed_bytes INTEGER, crc32 INTEGER, method INTEGER)')
    contact_records=[];csvs=[];samples=[];counts={};preview_counts={};contact_issues=[]
    with MultipartFile(parts) as source, zipfile.ZipFile(source) as archive:
        members=archive.infolist();total=len(members)
        present_streams=set()
        for member in members:
            fields=member.filename.split('/')
            if len(fields)>=4 and not member.is_dir():
                present_streams.add(tuple(fields[:3]))
        for i,member in enumerate(members):
            name=member.filename;safe_member(name)
            if member.is_dir():continue
            database.execute('INSERT INTO members VALUES(?,?,?,?,?,?)',
                (name,member.header_offset,member.file_size,member.compress_size,member.CRC,member.compress_type))
            suffix=Path(name).suffix.lower();counts[suffix]=counts.get(suffix,0)+1
            if suffix=='.csv' and 'contact' in Path(name).name.lower():
                if member.file_size>64*2**20:raise ValueError('Unexpectedly large contact metadata')
                data=archive.read(member)  # Checks CRC; retain exact released CSV.
                out=job.root/'contacts'/safe_member(name);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
                csvs.append(dict(member=name,path=str(out),sha256=file_hash(out)))
                from .acquisition_quality import parse_contact_csv
                folders=[row[0] for row in csv.reader(io.StringIO(data.decode('utf-8-sig'))) if row]
                subset={'contact_fine.csv':'data_fine','contact_indoor.csv':'data_indoor',
                        'contact_outdoor.csv':'data_outdoor'}.get(Path(name).name)
                if subset is None:raise ValueError('Unrecognized TacQuad contact layout')
                aliases={'gelsight_mini':'gelsight','digit':'digit','duragel':'duragel'}
                absent={(folder,sensor) for folder in folders for sensor,stream in aliases.items()
                        if (subset,folder,stream) not in present_streams}
                records,issues=parse_contact_csv(data,Path(name).name,absent_streams=absent)
                for record in records:record['csv_member']=name
                contact_records.extend(records);contact_issues.extend(issues)
            # Decode bounded native examples to audit shapes. No tactile pixel
            # resize, video conversion, or bulk image extraction is performed.
            if suffix in ('.jpg','.jpeg','.png'):
                parent=str(Path(name).parent)
                fields=parent.split('/')
                group=fields[0]+'/'+fields[2] if len(fields)>=3 else parent
                if preview_counts.get(group,0)<2 and len(samples)<64:
                    if member.file_size>32*2**20:raise ValueError('Unexpected native image size')
                    payload=archive.read(member)
                    with Image.open(io.BytesIO(payload)) as image:
                        image.load();shape=[image.height,image.width,len(image.getbands())]
                    samples.append(dict(member=name,shape=shape,sha256=hashlib.sha256(payload).hexdigest()))
                    preview_counts[group]=preview_counts.get(group,0)+1
            if (i+1)%5000==0:
                database.commit();job.report(stage='indexing_native_archive',completed_units=i+1,total_units=total)
        database.commit()
    database.close();temp.replace(dbpath)
    with (job.root/'contact_intervals.jsonl').open('w') as handle:
        for row in contact_records:handle.write(json.dumps(row)+'\n')
    write_json(job.root/'quality.json',dict(member_counts=counts,native_image_examples=samples,
        contacts=len(contact_records),framewise_alignment_verified=False,
        unavailable_sensor_intervals=sum(not valid for r in contact_records for valid in r['interval_valid'].values()),
        contact_annotation_issues=contact_issues,
        coarse_policy='spatial contact correspondence only; never infer temporal matches',
        fine_policy='released intervals retained; exact cross-sensor frame maps require an audited protocol',
        scene_ego_clip_compatibility_verified=False))
    write_json(job.root/'manifest.json',dict(schema='tacquad-archive-benchmark-v1',source='tacquad',
        revision=job.spec['revision'],archive_parts=[dict(path=str(p),sha256=file_hash(p)) for p in parts],
        member_index=str(dbpath),contact_csvs=csvs,contact_intervals=str(job.root/'contact_intervals.jsonl'),
        native_payloads_preserved=True,joint_source=False,excluded_from_joint_training=True))
    job.report(stage='indexed',completed_units=total,total_units=total)
    if contact_issues:
        from .acquisition_queue import QualityBlock
        raise QualityBlock(f'{len(contact_issues)} invalid annotation entries isolated; valid contact records indexed')


class TacQuadReader:
    """Read native benchmark members in place; keep open across a batch.

    The SQLite index supports dataset discovery. ZipFile checks each requested
    member's CRC; no image files or concatenated archive are persisted.
    """
    def __init__(self,manifest_path):
        manifest=json.loads(Path(manifest_path).read_text())
        if manifest.get('schema')!='tacquad-archive-benchmark-v1':
            raise ValueError('Expected TacQuad archive manifest')
        self.manifest=manifest
        self.source=MultipartFile([r['path'] for r in manifest['archive_parts']])
        self.archive=zipfile.ZipFile(self.source)

    def read(self,name):
        safe_member(name)
        return self.archive.read(name)

    def image(self,name):
        with Image.open(io.BytesIO(self.read(name))) as image:
            # Preserve native mode and dimensions; no RGB conversion of sensor
            # channels or 16-bit pressure-like payloads.
            image.load();return np.array(image)

    def array(self,name):
        return np.load(io.BytesIO(self.read(name)),allow_pickle=False)

    def close(self):
        self.archive.close();self.source.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
