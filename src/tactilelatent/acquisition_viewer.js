/* Native numeric sensor inspection. All scaling here changes the display only. */
window.NativeSensor = (() => {
 'use strict';
 const $ = id => document.getElementById(id);
 let active=false,current=null,dataset=null,api=null,graphRevision=0,graphAbort=null;
 const number=x=>x==null?'missing':Number(x).toLocaleString(undefined,{maximumSignificantDigits:9});
 function node(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;}
 function init(hooks){
  api=hooks.api;
  const style=node('style');style.textContent=`
   .native-toolbar{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}.native-toolbar>div{flex:1;min-width:160px}
   .native-charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
   .native-charts figure{margin:0}.native-charts figcaption{font-size:13px;margin:5px 0;color:#b9d5e2}
   .native-charts canvas,#nativeTrace{aspect-ratio:auto;image-rendering:auto;background:#0c121c}
   #nativeTrace{margin-top:10px}#nativeTable{max-height:340px;overflow:auto}#nativeValues summary{cursor:pointer;margin-top:12px}
   #nativeSummary{color:#85d9c7;font-size:13px;line-height:1.5}#nativeReadout{font:12px ui-monospace,monospace;min-height:36px;margin-top:10px}
   #nativeChannels{margin-top:12px;width:100%}
  `;document.head.append(style);
  const root=node('div');root.id='nativeSensor';root.hidden=true;
  const title=node('h2','Measured sensor values');title.style.fontSize='18px';
  const summary=node('p');summary.id='nativeSummary';
  const controls=node('div',null,'native-toolbar');
  for(const [id,label] of [['nativeGroup','Sensor / channel group'],['nativeAxis','XELA component (released axis order)'],['nativeScaleMode','Sensor display scale'],['nativeReferenceMode','Zero reference']]){
   const box=node('div'),l=node('label',label),select=node('select');l.htmlFor=id;select.id=id;box.append(l,select);controls.append(box);
  }
  $('outputPanel').insertBefore(root,$('outputPanel').querySelector('.maps'));
  root.append(title,summary,controls);
  const charts=node('div');charts.id='nativeCharts';charts.className='native-charts';root.append(charts);
  const readout=node('div');readout.id='nativeReadout';root.append(readout);
  const traceLabel=node('label','Channel dynamics · native units, gaps are missing observations');traceLabel.htmlFor='nativeChannels';
  const channels=node('select');channels.id='nativeChannels';const trace=node('canvas');trace.id='nativeTrace';trace.width=1000;trace.height=230;trace.setAttribute('aria-label','Selected sensor channel over time');
  const traceNote=node('p',null,'note');traceNote.id='nativeTraceNote';
  root.append(traceLabel,channels,trace,traceNote);
  const details=node('details');details.id='nativeValues';const heading=node('summary');heading.id='nativeValuesHeading';
  const table=node('div');table.id='nativeTable';details.append(heading,table);root.append(details);
  const note=node('p',null,'note');note.id='nativeScaleNote';root.insertBefore(note,charts);
  const zeroNote=node('p',null,'note');zeroNote.id='nativeZeroNote';root.insertBefore(zeroNote,charts);
  $('nativeGroup').onchange=()=>{draw();};$('nativeAxis').onchange=()=>{draw();};channels.onchange=drawTrace;
  $('nativeScaleMode').onchange=()=>{draw();drawTrace();};
  $('nativeReferenceMode').replaceChildren(new Option('Fixed sensor reference · all episodes','fixed'),new Option('Unchanged released readings','raw'));
  $('nativeReferenceMode').onchange=()=>{draw();drawTrace();};
 }
 function setup(d){
  active=!!d.measured_only;dataset={...d};current=null;++graphRevision;if(graphAbort)graphAbort.abort();
  $('nativeSensor').hidden=!active;
  for(const selector of ['.maps','.legend','.scales','.metrics'])$('outputPanel').querySelector(selector).hidden=active;
  $('model').disabled=active&&!d.reconstruction_available;
  if(!active)return;
  $('nativeControl').hidden=true;$('nativePanel').hidden=true;$('rawValues').replaceChildren();$('readout').textContent='';$('units').textContent='';
  $('inputNote').textContent='Scene RGB is paired using the recorded frame mapping. The plots show native measured sensor values; no reconstruction model has been trained for this source.';
  $('nativeSummary').textContent='Loading native measurements…';$('nativeCharts').replaceChildren();$('nativeTable').replaceChildren();
  $('nativeGroup').replaceChildren(new Option('All channel groups','all'),...d.groups.map((g,i)=>new Option(g.label,i)));
  if(d.episode_dependent_layout)$('nativeGroup').replaceChildren(new Option('Loading episode hand mapping…','all'));
  dataset.layoutKey=null;
  $('nativeAxis').replaceChildren(...[0,1,2].map(i=>new Option('Axis '+i,i)));$('nativeAxis').parentElement.hidden=d.id!=='tdex';
  $('nativeScaleMode').replaceChildren(...(d.display_policy?[new Option('FEEL detail · fixed 0–'+number(d.display_policy.detail_upper),'detail')]:[]),new Option('Full training range · all episodes','fixed'),new Option('Shared quantity · this episode','shared'),new Option('Per channel · magnifies small fluctuations','channel'));
  $('nativeReferenceMode').value='fixed';$('nativeZeroNote').textContent='Loading fixed-reference status…';
  $('nativeChannels').replaceChildren(...d.labels.map((label,i)=>new Option(label,i)));
  if(d.episode_dependent_layout)$('nativeChannels').replaceChildren();
  $('nativeReadout').textContent='Hover over a bar for its exact native value.';
 }
 function shownIndices(group){return group.indices.filter(i=>dataset.id!=='tdex'||i%3===Number($('nativeAxis').value));}
 function reference(){return current?.acquisition.zero_reference;}
 function offset(index){return $('nativeReferenceMode').value==='fixed'?(reference()?.offset[index]||0):0;}
 function value(index,raw=current.original[index]){return raw==null?null:raw-offset(index);}
 function band(index){const r=reference(),b=r?.unloaded_envelope?.[index];return b?b.map(v=>v+r.offset[index]-offset(index)):null;}
 function positiveOnly(){return (dataset.id==='feel'||dataset.id.startsWith('feel_'))&&current?.acquisition.display_policy?.nonnegative===true;}
 function sharedScale(){return ($('nativeScaleMode').value==='detail'&&current?.acquisition.display_policy?Array(current.acquisition.channels).fill(current.acquisition.display_policy.detail_upper):null)||($('nativeScaleMode').value==='fixed'&&reference()?.display_limits)||($('nativeScaleMode').value==='shared'&&(current?.acquisition.shared_display_scales||current?.acquisition.shared_wrench_scales));}
 function scales(){return sharedScale()||current.acquisition.display_scales;}
 function draw(){
  if(!active||!current)return;
  const data=current,info=data.acquisition,selected=$('nativeGroup').value;
  const limits=scales(),shared=sharedScale();
  let rangeNote='Each channel fills its own episode range. This magnifies small fluctuations; bar lengths across channels are not comparable.';
  if(shared){
   if(dataset.id==='trex')rangeNote='Force components share ±'+number(limits[0])+'; moment components share ±'+number(limits[3])+' across both hands (different native units).';
   else if(dataset.id==='hrdexdb_inspire_f1')rangeNote='Across all eight sensors: normal force ±'+number(limits[0])+'; tangential force ±'+number(limits[1])+'; direction ±'+number(limits[2])+' (separate native quantities). Direction is not force; raw direction values, including −1, are preserved.';
   else rangeNote='All four released tactile channels share ±'+number(limits[0])+' in native units; this is not calibrated SI force.';
   rangeNote+=' Bars and the trace use these fixed episode ranges.';
  }
  if($('nativeScaleMode').value==='fixed'&&reference())rangeNote='One training-derived range for every episode. '+(dataset.id==='trex'?'Force ±'+number(limits[0])+'; moment ±'+number(limits[3])+'.':dataset.id==='hrdexdb_inspire_f1'?'Normal ±'+number(limits[0])+'; tangential ±'+number(limits[1])+'; direction ±'+number(limits[2])+'.':dataset.id==='tdex'?'Axis 0 ±'+number(limits[0])+'; axis 1 ±'+number(limits[1])+'; axis 2 ±'+number(limits[2])+'.':'All force/contact channels ±'+number(limits[0])+'.');
  if(positiveOnly()){
   if($('nativeScaleMode').value==='detail')rangeNote='One training-derived range for every FEEL episode: 0–'+number(limits[0])+', shared across both hands and Lab/Kitchen. Zero stays fixed at the bottom. Rare larger readings are marked ▲ with their actual values; select Full training range to see their complete height.';
   else if($('nativeScaleMode').value==='fixed')rangeNote='Full training range for this FEEL layout: 0–'+number(limits[0])+', shared across channels and episodes. Rare peaks make ordinary changes look small; FEEL detail uses the same fixed zero with a closer range.';
   else if($('nativeScaleMode').value==='shared')rangeNote='All FEEL channels share 0–'+number(limits[0])+' for this episode. Zero remains fixed; heights cannot be compared with another episode unless the range matches.';
   else rangeNote='Each FEEL channel fills its own episode range from zero. This magnifies small fluctuations; bar heights are not comparable across channels or episodes.';
  }
  $('nativeScaleNote').textContent=rangeNote+' Measured: teal/red. Reconstruction: orange. Gray is missing. Values beyond the displayed range are marked ▲. Raw readings and stored training targets are unchanged.';
  $('nativeZeroNote').textContent=reference()?reference().status+'. '+reference().explanation+(reference().unloaded_envelope?' Amber = observed unloaded range in screened training frames; this is not a calibrated confidence interval or contact threshold.':'')+' No per-episode refit or forced zero threshold.':'No matching fixed reference for this release/layout. Showing unchanged native readings; numeric 0 is not a verified physical no-contact calibration.';
  $('nativeReferenceMode').disabled=!reference();
  $('nativeCharts').replaceChildren();const visible=[];
  info.groups.forEach((group,g)=>{
   if(selected!=='all'&&Number(selected)!==g)return;const indices=shownIndices(group);visible.push(...indices);
   const figure=node('figure'),caption=node('figcaption',group.label),canvas=node('canvas');canvas.width=640;canvas.height=190;canvas.setAttribute('aria-label',group.label+' native signed readings');figure.append(caption,canvas);$('nativeCharts').append(figure);
   const ctx=canvas.getContext('2d'),left=positiveOnly()?42:22,right=12,top=positiveOnly()?25:10,bottom=33,mid=positiveOnly()?canvas.height-bottom:(canvas.height-bottom+top)/2,span=positiveOnly()?canvas.height-bottom-top:(canvas.height-bottom-top)/2,w=(canvas.width-left-right)/indices.length;
   ctx.fillStyle='#0c121c';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#78879b';ctx.beginPath();ctx.moveTo(left,mid);ctx.lineTo(canvas.width-right,mid);ctx.stroke();
   ctx.font='15px system-ui';ctx.fillStyle='#a8b5c6';ctx.fillText('0',3,mid+5);
   if(positiveOnly()){ctx.font='12px system-ui';ctx.fillText(number(limits[indices[0]]),3,top+4,35);}
   indices.forEach((index,j)=>{
    const x=left+j*w,y=value(index),scale=limits[index],envelope=band(index);
    if(envelope){ctx.fillStyle='rgba(245,183,66,.3)';const upper=mid-Math.max(-1,Math.min(1,envelope[1]/scale))*span,lower=mid-Math.max(-1,Math.min(1,envelope[0]/scale))*span;ctx.fillRect(x+2,upper,Math.max(1,w-4),Math.max(2,lower-upper));}
    ctx.fillStyle=y==null?'#697483':y<0?'#ff8888':'#56cbb6';
    const height=y==null?6:Math.min(1,Math.abs(y)/scale)*span;
    const hasPrediction=Array.isArray(data.reconstruction),barWidth=hasPrediction?w*.42:Math.max(1,w-6);
    if(!positiveOnly()||y!==0)ctx.fillRect(x+Math.min(3,w/8),y==null?mid-3:y>=0?mid-height:mid,barWidth,Math.max(y===0?2:1,height));
    const reconstructed=hasPrediction?value(index,data.reconstruction[index]):null;
    if(hasPrediction){const ph=reconstructed==null?6:Math.min(1,Math.abs(reconstructed)/scale)*span;ctx.fillStyle=reconstructed==null?'#697483':'#ffb454';if(!positiveOnly()||reconstructed!==0)ctx.fillRect(x+w*.53,reconstructed==null?mid-3:reconstructed>=0?mid-ph:mid,w*.42,Math.max(1,ph));if(reconstructed!=null&&Math.abs(reconstructed)>scale){ctx.font='14px system-ui';ctx.fillText('▲',x+w*.6,reconstructed>0?top+12:canvas.height-bottom-3);}}
    ctx.fillStyle='#afbdd0';ctx.textAlign='center';const short=dataset.id==='tdex'?'T'+Math.floor((index%48)/3):data.labels[index].split('_').slice(-2).join('_');
    ctx.fillText(short,x+w/2,canvas.height-10,Math.max(10,w-3));
    if(positiveOnly()){
     ctx.font='16px ui-monospace,monospace';ctx.fillStyle=y==null?'#a8b5c6':Math.abs(y)>scale?'#ffcf77':'#e6edf5';
     const text=(y==null?'missing':(Math.abs(y)>scale?'▲ ':'')+Number(y).toLocaleString(undefined,{maximumSignificantDigits:4}))+(hasPrediction?' | '+(reconstructed==null?'missing':Number(reconstructed).toLocaleString(undefined,{maximumSignificantDigits:3})): '');
     ctx.fillText(text,x+w/2,y==null?mid-12:Math.max(17,mid-height-7),Math.max(10,w-5));ctx.font='15px system-ui';
    }else if(y!=null&&Math.abs(y)>scale){ctx.fillStyle='#ffcf77';ctx.fillText('▲',x+w/2,y>0?top+12:canvas.height-bottom-3);}
   });
   canvas.onmousemove=event=>{const box=canvas.getBoundingClientRect(),j=Math.floor(((event.clientX-box.left)*canvas.width/box.width-left)/w);if(j<0||j>=indices.length)return;const i=indices[j];$('nativeReadout').textContent=data.labels[i]+' · shown '+number(value(i))+' · raw '+number(data.original[i])+(data.reconstruction?' · AE '+number(value(i,data.reconstruction[i]))+' · error '+number(data.error[i]):'')+' · fixed offset '+number(reference()?.offset[i]||0)+' · native '+info.native_dtype;};
  });
  $('nativeValuesHeading').textContent='Exact native and displayed values · '+visible.length+' channels';const table=node('table'),header=table.insertRow();for(const label of ['Released channel','Raw value','Displayed value','AE reconstruction','Absolute error','Fixed offset','Observation'])header.append(node('th',label));
  visible.forEach(i=>{const row=table.insertRow();for(const text of [data.labels[i],number(data.original[i]),number(value(i)),data.reconstruction?number(value(i,data.reconstruction[i])):'—',data.error?number(data.error[i]):'—',number(reference()?.offset[i]||0),data.original[i]==null?'Missing':'Valid'])row.insertCell().textContent=text;});$('nativeTable').replaceChildren(table);
 }
 function render(data){
  if(!active||!data.acquisition)return;
  const changed=!current||current.episode!==data.episode||current.frame!==data.frame||current.acquisition.reconstruction?.model_id!==data.acquisition.reconstruction?.model_id||current.acquisition.reconstruction?.epoch!==data.acquisition.reconstruction?.epoch;current=data;const info=data.acquisition;
  dataset.direct_prediction_only=!!info.direct_prediction_only;
  const layoutKey=JSON.stringify([data.labels,info.groups]);
  if(dataset.layoutKey!==layoutKey){
   dataset.layoutKey=layoutKey;
   dataset.labels=data.labels;dataset.groups=info.groups;
   $('nativeGroup').replaceChildren(new Option('All channel groups','all'),...info.groups.map((g,i)=>new Option(g.label,i)));
   $('nativeChannels').replaceChildren(...data.labels.map((label,i)=>new Option(label,i)));
  }
  $('nativeSummary').textContent=info.valid_count+' / '+info.channels+' valid channels · native shape '+data.target_shape.join(' × ')+' · '+info.native_dtype+(info.reconstruction?.kind==='visual'?' · visual prediction · native MAE '+number(info.reconstruction.mae):info.reconstruction?' · '+info.reconstruction.code_dim+'D AE · epoch '+info.reconstruction.epoch+' · mean native channel error '+number(info.reconstruction.mae)+(info.reconstruction.accepted?'':' · baseline quality check pending'):dataset.direct_prediction_only?' · direct prediction targets · no autoencoder':' · awaiting saved teacher for this layout');
  $('inputNote').textContent=info.reconstruction?.kind==='visual'?'The joint visual model predicts the current tactile reading using this frame and seven history frames. Orange bars show its prediction. The channel history shows measured values.':info.reconstruction?'The autoencoder reconstructs measured tactile values at this frame. Scene RGB provides context and is not an input to this reconstruction. Orange bars/trace show the reconstruction.':dataset.direct_prediction_only?'The '+info.channels+' native tactile measurements are direct prediction targets. No autoencoder is used. Scene RGB is shown for context.':'Scene RGB is paired using the recorded frame mapping. No saved reconstruction is selected for this layout.';
  if(info.sensor_layout){const layout=info.sensor_layout;$('nativeSummary').textContent+=' · '+(layout.measured_hands.length?layout.measured_hands.map(h=>h+' hand').join(' + ')+' measured':'hand identity unverified')+(layout.unmeasured_hands.length?' · '+layout.unmeasured_hands.map(h=>h+' hand').join(' + ')+' not recorded':'');}
  $('sensorDetails').textContent=(info.units||'Native released readings')+' · target '+data.target_shape.join(' × ');
  $('timestamp').textContent=info.elapsed_seconds.toFixed(3)+' s · native frame '+info.source_frame_id;
  $('clipDetails').hidden=false;$('clipDetails').textContent='Recorded RGB indices: '+Object.entries(info.camera_frame_indices).map(([camera,index])=>camera+': '+(index==null?'missing':index)).join(' · ')+(info.alignment_error_seconds==null?'':' · pairing offset '+(1000*info.alignment_error_seconds).toFixed(2)+' ms');
  draw();if(changed)drawTrace();
 }
 async function drawTrace(){
  if(!active||!current)return;const token=++graphRevision;if(graphAbort)graphAbort.abort();graphAbort=new AbortController();
  const channel=$('nativeChannels').value,data=current;$('nativeTraceNote').textContent='Loading channel history…';
  try{
   const response=await fetch(api('acquisition/series?'+new URLSearchParams({episode:data.episode,frame:data.frame,channel,model:data.acquisition.reconstruction?.model_id??-1})),{signal:graphAbort.signal});const series=await response.json();if(!response.ok)throw Error(series.error);if(token!==graphRevision||!active)return;
   const shown=series.values.map(v=>value(Number(channel),v)),canvas=$('nativeTrace'),ctx=canvas.getContext('2d'),values=shown.filter(x=>x!=null),times=series.seconds;ctx.fillStyle='#0c121c';ctx.fillRect(0,0,canvas.width,canvas.height);
   if(!values.length){$('nativeTraceNote').textContent='No valid observations in this window';return;}
   let low=Math.min(...values),high=Math.max(...values);if(sharedScale()){const range=scales()[Number(channel)];low=positiveOnly()?0:-range;high=range;}else{if(low===high){low-=Math.max(1,Math.abs(low)*.05);high+=Math.max(1,Math.abs(high)*.05);}const pad=(high-low)*.08;low=positiveOnly()?Math.min(0,low):low-pad;high+=pad;}
   const x=t=>85+(t-times[0])/Math.max(1e-12,times.at(-1)-times[0])*(canvas.width-110),y=v=>15+(high-v)/(high-low)*(canvas.height-50);
   const envelope=band(Number(channel));if(envelope){const top=y(Math.min(high,envelope[1])),bottom=y(Math.max(low,envelope[0]));ctx.fillStyle='rgba(245,183,66,.25)';if(bottom>=top)ctx.fillRect(85,top,canvas.width-110,Math.max(2,bottom-top));}
   ctx.font='15px system-ui';ctx.fillStyle='#afbdd0';ctx.fillText(number(high),4,25,78);ctx.fillText(number(low),4,canvas.height-35,78);ctx.fillText(times[0].toFixed(2)+' s',85,canvas.height-8);ctx.fillText(times.at(-1).toFixed(2)+' s',canvas.width-100,canvas.height-8);
   ctx.save();ctx.beginPath();ctx.rect(85,15,canvas.width-110,canvas.height-50);ctx.clip();ctx.strokeStyle='#78879b';ctx.beginPath();ctx.moveTo(85,y(0));ctx.lineTo(canvas.width-25,y(0));ctx.stroke();
   ctx.strokeStyle='#56cbb6';ctx.lineWidth=2;ctx.beginPath();let started=false;shown.forEach((v,i)=>{if(v==null){started=false;return;}if(started)ctx.lineTo(x(times[i]),y(v));else ctx.moveTo(x(times[i]),y(v));started=true;});ctx.stroke();ctx.restore();
   if(series.reconstruction){ctx.save();ctx.beginPath();ctx.rect(85,15,canvas.width-110,canvas.height-50);ctx.clip();ctx.strokeStyle='#ffb454';ctx.lineWidth=2;ctx.beginPath();let began=false;series.reconstruction.forEach((raw,i)=>{const v=value(Number(channel),raw);if(v==null){began=false;return;}if(began)ctx.lineTo(x(times[i]),y(v));else ctx.moveTo(x(times[i]),y(v));began=true;});ctx.stroke();ctx.restore();}
   ctx.fillStyle='#ffcf77';shown.forEach((v,i)=>{if(v==null||v>=low&&v<=high)return;const px=x(times[i]),py=v>high?15:canvas.height-35;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(px-4,py+(v>high?7:-7));ctx.lineTo(px+4,py+(v>high?7:-7));ctx.closePath();ctx.fill();});
   const selected=series.frame_indices.indexOf(data.frame);ctx.strokeStyle='#ff8888';ctx.beginPath();ctx.moveTo(x(times[selected]),8);ctx.lineTo(x(times[selected]),canvas.height-35);ctx.stroke();
   $('nativeTraceNote').textContent=series.label+' · measured '+number(value(Number(channel)))+' · raw '+number(data.original[Number(channel)])+(series.reconstruction?' · reconstruction '+number(value(Number(channel),data.reconstruction[Number(channel)]))+' · teal: measured · orange: reconstruction':'')+' · red line: current frame · '+(envelope?'amber: observed unloaded training range · ':'')+'window ±150 observations'+(values.some(v=>v<low||v>high)?' · some values exceed the fixed display range':'');
  }catch(error){if(error.name!=='AbortError'&&token===graphRevision)$('nativeTraceNote').textContent='History unavailable: '+error.message;}
 }
 return {init,setup,render};
})();
