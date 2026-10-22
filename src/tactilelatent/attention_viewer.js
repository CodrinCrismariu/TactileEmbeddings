'use strict';
let attentionData={},attentionImages={},attentionController,attentionRevision=0;

function attentionSupported(suffix='') { const run=selectedRun(suffix);return !!run?.attention_supported&&(!run.attention_sources||run.attention_sources.includes(selectedDataset().id)); }
function updateAttentionAvailability() {
 if(!jointMode)return;
 $('attentionToggle').hidden=false;
 const supported=attentionSupported()||($('modelB').value!=='-1'&&attentionSupported('B'));
 $('showAttention').disabled=!supported;
 $('attentionHint').textContent=supported?'Inspect the tactile query’s attention to scene regions.':selectedRun()?.attention_supported?'Select a model with a trained head for this dataset. The 17-source model supports EgoPressure and the new datasets.':'Select a spatial-attention V-JEPA model to inspect attention.';
 $('attentionPanel').hidden=!supported||!$('showAttention').checked;
}
function resetAttention() {
 attentionRevision++;attentionController?.abort();attentionData={};
 for(const image of Object.values(attentionImages))image.close?.();
 attentionImages={};
 if(jointMode&&$('showAttention').checked) {
  $('attentionStatus').textContent='Loading attention for the selected frame…';
  for(const suffix of ['','B']) {
   const canvas=$('attentionCanvas'+suffix);canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
   $('attentionReadout'+suffix).textContent='';
  }
 }
}
function replaceAttentionOptions(id,options,fallback) {
 const select=$(id),old=select.value;
 select.replaceChildren(...options.map(([label,value])=>new Option(label,String(value))));
 select.value=options.some(([,value])=>String(value)===old)?old:String(fallback);
}
async function loadAttention(frameRevision=revision) {
 if(!jointMode)return;
 updateAttentionAvailability();
 if($('attentionPanel').hidden)return;
 resetAttention();const mine=attentionRevision;
 attentionController=new AbortController();const signal=attentionController.signal;
 const suffixes=$('modelB').value==='-1'?['']:['','B'];
 $('attentionFigureB').hidden=suffixes.length===1;
 const requests=suffixes.map(async suffix=>{
  const data=suffix?currentDataB:currentData,run=selectedRun(suffix);
  const label=suffix?'Model B':'Model A';
  if(!attentionSupported(suffix)||!data?.joint)return {suffix,error:'Spatial attention is unavailable for this selection.'};
  const params=new URLSearchParams({episode:$('episode').value,frame,model:run.id,
   epoch:$('epoch'+suffix).value,views:$('views').value,expected_epoch:data.joint.epoch});
  try {
   const result=await readJson(api('joint/attention?'+params),signal);
   const bytes=Uint8Array.from(atob(result.weights_b64),c=>c.charCodeAt(0));
   const view=new DataView(bytes.buffer),values=new Float32Array(bytes.byteLength/4);
   for(let i=0;i<values.length;i++)values[i]=view.getFloat32(i*4,true);
   return {suffix,result:{...result,values,request:params,label,dimension:run.config.latent_dim}};
  } catch(error) { if(error.name==='AbortError')throw error;return {suffix,error:error.message}; }
 });
 try {
  const results=await Promise.all(requests);
  if(mine!==attentionRevision||frameRevision!==revision)return;
  for(const item of results) {
   $('attentionTitle'+item.suffix).textContent=item.result?`${item.result.label} · ${item.result.dimension}D · epoch ${item.result.epoch}`:(item.suffix?'Model B':'Model A');
   if(item.result)attentionData[item.suffix]=item.result;
   else $('attentionReadout'+item.suffix).textContent=item.error;
  }
  const maps=Object.values(attentionData);
  if(!maps.length){$('attentionStatus').textContent='No attention maps available for these selections.';return;}
  const primary=maps[0],episode=catalog.episodes.find(e=>e.id===Number($('episode').value));
  const slots=[...new Set(maps.flatMap(m=>m.camera_slots))].sort();
  replaceAttentionOptions('attentionCamera',slots.map(slot=>[episode?.camera_labels?.[slot]||{'0':'Main / ego','1':'Left / scene','2':'Right wrist'}[slot]||slot,slot]),slots[0]);
  replaceAttentionOptions('attentionHistory',[['Target image','target'],...primary.clip_frames.map((value,i)=>[`Input ${i+1} · frame ${value}`,i])],'target');
  const layers=Math.min(...maps.map(m=>m.shape[0])),heads=Math.min(...maps.map(m=>m.shape[1]));
  replaceAttentionOptions('attentionLayer',[['Last attention layer','last'],...Array.from({length:layers},(_,i)=>['Layer '+(i+1),i])],'last');
  replaceAttentionOptions('attentionHead',[['Mean of heads','mean'],...Array.from({length:heads},(_,i)=>['Head '+(i+1),i])],'mean');
  $('attentionStatus').textContent='Yellow/red regions receive more attention. Adjust image brightness and heatmap opacity for visibility. Color scale is shared across Models A and B.';
  await loadAttentionImages(mine,frameRevision);
 } catch(error) {
  if(error.name!=='AbortError'&&mine===attentionRevision)$('attentionStatus').textContent=error.message;
 }
}
function attentionImageSlot(map) {return $('attentionHistory').value==='target'?map.default_image_slot:Number($('attentionHistory').value);}
async function loadAttentionImages(mine=attentionRevision,frameRevision=revision) {
 const camera=$('attentionCamera').value;
 await Promise.all(Object.entries(attentionData).map(async ([suffix,map])=>{
  const slot=attentionImageSlot(map),key=`${suffix}:${camera}:${slot}`;
  if(!map.camera_slots.includes(camera))return;
  if(!attentionImages[key]) {
   const params=new URLSearchParams(map.request);params.delete('expected_epoch');params.set('camera',camera);params.set('history',slot);
   try {
    const response=await fetch(api('joint/attention-image?'+params),{signal:attentionController.signal});
    if(!response.ok)throw new Error((await response.json()).error);
    const image=await createImageBitmap(await response.blob());
    if(mine!==attentionRevision||frameRevision!==revision){image.close();return;}
    attentionImages[key]=image;
   } catch(error) {if(error.name!=='AbortError'&&mine===attentionRevision)$('attentionReadout'+suffix).textContent=error.message;}
  }
 }));
 if(mine===attentionRevision&&frameRevision===revision)paintAttention();
}
function selectedAttentionMap(map) {
 const [layers,heads,groups,views,gy,gx]=map.shape;
 const layer=$('attentionLayer').value==='last'?layers-1:Number($('attentionLayer').value);
 const chosen=$('attentionHead').value==='mean'?Array.from({length:heads},(_,i)=>i):[Number($('attentionHead').value)];
 const count=groups*views*gy*gx,values=new Float32Array(count);
 for(const head of chosen)for(let i=0;i<count;i++)values[i]+=map.values[(layer*heads+head)*count+i]/chosen.length;
 let peak=0;for(const value of values)peak=Math.max(peak,value);
 return {values,peak,layer,gy,gx};
}
function paintAttention() {
 if(!jointMode||$('attentionPanel').hidden)return;
 const camera=$('attentionCamera').value,opacity=Number($('attentionOpacity').value)/100;
 const brightness=Number($('attentionBrightness').value)/100;
 $('attentionOpacityValue').textContent=Math.round(opacity*100)+'%';
 $('attentionBrightnessValue').textContent=Math.round(brightness*100)+'%';
 const selected=Object.fromEntries(Object.entries(attentionData).map(([suffix,map])=>[suffix,selectedAttentionMap(map)]));
 const peak=Math.max(1e-12,...Object.values(selected).map(m=>m.peak));
 const colors=[[20,40,140],[0,220,255],[255,255,60],[255,120,20],[255,40,20]];
 for(const [suffix,map] of Object.entries(attentionData)) {
  const canvas=$('attentionCanvas'+suffix),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  const slot=attentionImageSlot(map),image=attentionImages[`${suffix}:${camera}:${slot}`];
  if(!map.camera_slots.includes(camera)){$('attentionReadout'+suffix).textContent='This camera was not used by this model.';continue;}
  if(!image)continue;
  ctx.drawImage(image,0,0,canvas.width,canvas.height);
  // Dim only the displayed RGB; retain the original model input and weights.
  ctx.fillStyle=`rgba(0,0,0,${1-brightness})`;ctx.fillRect(0,0,canvas.width,canvas.height);
  const {values,gy,gx}=selected[suffix],group=Math.floor(slot/2),offset=(group*3+Number(camera))*gy*gx;
  const tile=document.createElement('canvas');tile.width=gx;tile.height=gy;
  const tileContext=tile.getContext('2d'),pixels=tileContext.createImageData(gx,gy);let mass=0;
  for(let i=0;i<gy*gx;i++) {
   const value=values[offset+i],strength=Math.sqrt(Math.max(0,Math.min(1,value/peak)));mass+=value;
   const position=strength*(colors.length-1),index=Math.min(colors.length-2,Math.floor(position)),mix=position-index;
   for(let channel=0;channel<3;channel++)pixels.data[i*4+channel]=Math.round(colors[index][channel]*(1-mix)+colors[index+1][channel]*mix);
   pixels.data[i*4+3]=Math.round(255*opacity*strength);
  }
  tileContext.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(tile,0,0,canvas.width,canvas.height);
  $('attentionReadout'+suffix).textContent=`Image frame ${map.clip_frames[slot]} · temporal pair ${map.temporal_groups[group].join(', ')} · ${(100*mass).toFixed(1)}% of total attention in this camera/pair`;
 }
}
function initializeAttentionControls() {
 if(!jointMode)return;
 $('attentionToggle').hidden=false;
 $('showAttention').onchange=()=>{pause();if($('showAttention').checked)loadAttention();else{resetAttention();updateAttentionAvailability();}};
 for(const name of ['attentionCamera','attentionHistory'])$(name).onchange=()=>loadAttentionImages();
 for(const name of ['attentionLayer','attentionHead'])$(name).onchange=paintAttention;
 for(const name of ['attentionOpacity','attentionBrightness'])$(name).oninput=paintAttention;
}
