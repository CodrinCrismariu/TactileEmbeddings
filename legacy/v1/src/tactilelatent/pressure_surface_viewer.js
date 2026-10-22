/* EgoPressure display only. Native targets and quantitative errors are untouched. */
'use strict';
window.PressureSurface=(()=>{
 const $=id=>document.getElementById(id), kinds=['original','reconstruction','error'];
 const palette=[[5,10,20],[20,123,157],[27,180,189],[154,219,103],[255,241,155]];
 let hooks,assets,current,episode,active=false,sequence=0,abort,pose=null,poseKey='',expanded=false;
 let mode='atlas',preferredMode='mesh',yaw=Math.PI,pitch=0,zoom=1,uvCursor=null;
 let geometryCache=null;
 const renderers=new Map(),poseCache=new Map();
 const gamma=()=>Number($('pressureContrast').value);
 function color(value,maximum){
  if(value===null||!Number.isFinite(value))return [65,72,84];
  const t=Math.pow(Math.max(0,Math.min(1,value/maximum)),gamma())*4;
  const i=Math.min(3,Math.floor(t)),f=t-i;
  return palette[i].map((a,k)=>Math.round(a+(palette[i+1][k]-a)*f));
 }
 function atlas(canvas,values,maximum){
  const ctx=canvas.getContext('2d'),W=224,H=224,image=ctx.createImageData(W,H);
  for(let i=0;i<W*H;i++)image.data.set([...color(values?.[i]??null,maximum),255],i*4);
  ctx.putImageData(image,0,0);canvas.pressurePayload={values,maximum};
  if($('pressureGrid').checked){
   ctx.strokeStyle='rgba(190,205,225,.28)';ctx.lineWidth=.4;
   for(let n=32;n<224;n+=32){ctx.beginPath();ctx.moveTo(n,0);ctx.lineTo(n,224);ctx.moveTo(0,n);ctx.lineTo(224,n);ctx.stroke();}
  }
  if(uvCursor){
   const [x,y]=uvCursor;ctx.strokeStyle='rgba(255,255,255,.85)';ctx.lineWidth=.5;
   ctx.beginPath();ctx.moveTo(x+.5,0);ctx.lineTo(x+.5,224);ctx.moveTo(0,y+.5);ctx.lineTo(224,y+.5);ctx.stroke();
  }
  if(!canvas.dataset.pressureBound){
   canvas.dataset.pressureBound='true';
   canvas.addEventListener('mousemove',event=>{
    if(!active)return;const r=canvas.getBoundingClientRect();
    uvCursor=[Math.max(0,Math.min(223,Math.floor((event.clientX-r.left)/r.width*224))),
              Math.max(0,Math.min(223,Math.floor((event.clientY-r.top)/r.height*224)))];
    redrawAtlases();
   });
   canvas.addEventListener('mouseleave',()=>{uvCursor=null;if(active)redrawAtlases();});
  }
 }
 function redrawAtlases(){
  for(const kind of kinds){const c=$(kind+'0');if(c?.pressurePayload)atlas(c,c.pressurePayload.values,c.pressurePayload.maximum);}
 }
 function setExpanded(value){
  expanded=value&&active;$('outputPanel').classList.toggle('pressure-expanded',expanded);
  document.body.classList.toggle('pressure-modal-open',expanded);
  $('pressureExpand').textContent=expanded?'Close enlarged view':'Enlarge comparison';
  $('pressureExpand').setAttribute('aria-expanded',String(expanded));
  $('pressureExpandedTimeline').hidden=!expanded;$('pressureContext').hidden=!expanded;
  if(expanded){$('outputPanel').setAttribute('role','dialog');$('outputPanel').setAttribute('aria-label','EgoPressure comparison');}
  else{$('outputPanel').removeAttribute('role');$('outputPanel').removeAttribute('aria-label');}
  drawMeshes();
 }
 function missingAssetsNote(side){
  if(assets?.sides?.[side]?.error)return 'The supplied UV mesh could not be validated: '+assets.sides[side].error;
  return '3D hand requires the official MANO UV mesh files. Place MANO_UV_left.obj and MANO_UV_right.obj in mano_models/ to enable it. The native UV atlas is available now.';
 }
 function configureMode(){
  const available=!!assets?.sides?.[current?.hand_side]?.available;
  const select=$('pressureViewMode');
  select.replaceChildren(new Option('UV atlas','atlas'));
  if(available)select.add(new Option('3D hand surface','mesh'));
  mode=available?preferredMode:'atlas';select.value=mode;
  $('pressureViewChoice').hidden=!available;
  $('pressureAssetNote').textContent=available?
   '3D uses the recorded hand pose for every panel; pressure comes from the selected measured or reconstructed UV map.':missingAssetsNote(current?.hand_side);
  $('pressureAssetLink').href=assets?.setup_url||'https://github.com/eth-siplab/EgoPressure/blob/master/mano_models/README.md';
  $('pressureAssetLink').hidden=available;
  showMode();
 }
 function showMode(){
  const mesh=active&&mode==='mesh'&&pose?.available;
  $('pressureMeshGrid').hidden=!mesh;$('outputPanel').querySelector('.maps').hidden=mesh;
  $('pressureOrbitControls').hidden=!(active&&mode==='mesh');
  if(active&&mode==='mesh')$('pressureSurfaceStatus').textContent=pose?.available?
   `${current.hand_side} hand · recorded pose · native frame ${pose.source_frame_id} · drag to rotate; scroll to zoom`:
   (pose?.reason||'Loading recorded hand pose…')+' · UV atlas shown';
  else $('pressureSurfaceStatus').textContent=active?
   '224 × 224 native UV atlas · shared pressure scale · brighter low values for inspection · hover for exact values':'';
  if(mesh)drawMeshes();
 }
 async function refreshAssets(){
  try{const response=await fetch(hooks.api('egopressure/surface-assets'));assets=await response.json();if(!response.ok)throw Error(assets.error);}
  catch(error){assets=null;if(active)$('pressureAssetNote').textContent='Hand asset check failed: '+error.message;return;}
  if(active){configureMode();if(mode==='mesh')loadPose();}
 }
 async function loadPose(){
  if(!active||mode!=='mesh'||!current)return;
  const key=episode+':'+current.frame;
  if(key===poseKey&&pose){showMode();return;}
  const token=++sequence;if(abort)abort.abort();abort=new AbortController();
  pose=null;poseKey='';showMode();
  try{
   let value=poseCache.get(key);
   if(!value){
    const response=await fetch(hooks.api('egopressure/surface?'+new URLSearchParams({episode,frame:current.frame})),{signal:abort.signal});
    value=await response.json();if(!response.ok)throw Error(value.error||'Pose request failed');
   }
   if(token!==sequence||!active||mode!=='mesh')return;
   if(value.frame!==current.frame||value.source_frame_id!==current.source_frame_id||value.hand_side!==current.hand_side)
    throw Error('Pose and tactile frame identifiers disagree');
   pose=value;poseKey=key;poseCache.set(key,value);if(poseCache.size>24)poseCache.delete(poseCache.keys().next().value);
   showMode();
  }catch(error){if(error.name==='AbortError'||token!==sequence)return;pose={available:false,reason:error.message};showMode();}
 }
 function setup(dataset){
  active=dataset?.id==='egopressure';$('pressureTools').hidden=!active;
  if(!active){++sequence;if(abort)abort.abort();pose=null;poseKey='';current=null;mode='atlas';setExpanded(false);}
  $('pressureMeshGrid').hidden=true;$('outputPanel').querySelector('.maps').hidden=false;
  if(active)configureMode();
 }
 function clear(){
  ++sequence;if(abort)abort.abort();current=null;pose=null;poseKey='';geometryCache=null;
  if(active)showMode();
 }
 function update(data,episodeId){
  if(!active||data?.source!=='egopressure')return;
  if(poseKey&&poseKey!==episodeId+':'+data.frame){++sequence;if(abort)abort.abort();pose=null;poseKey='';}
  current=data;episode=episodeId;
  $('pressureContext').src=$('video').src;
  $('pressureFrameSlider').max=$('frameSlider').max;$('pressureFrameSlider').value=data.frame;
  $('pressureFrameLabel').textContent='Frame '+data.frame+' / '+$('frameSlider').max;
  configureMode();if(mode==='mesh')loadPose();
 }
 function init(options){
  hooks=options;
  $('pressureExpand').onclick=()=>setExpanded(!expanded);
  $('pressureRefreshAssets').onclick=refreshAssets;
  $('pressureViewMode').onchange=()=>{hooks.pause();mode=preferredMode=$('pressureViewMode').value;if(mode==='mesh')loadPose();else{++sequence;if(abort)abort.abort();showMode();}};
  $('pressureContrast').oninput=()=>{$('pressureContrastValue').textContent=gamma().toFixed(2);hooks.redraw();drawMeshes();};
  $('pressureGrid').onchange=redrawAtlases;
  $('pressureResetView').onclick=()=>{yaw=Math.PI;pitch=0;zoom=1;drawMeshes();};
  $('pressureTurnOver').onclick=()=>{yaw+=Math.PI;drawMeshes();};
  $('pressureFrameSlider').oninput=event=>{$('frameSlider').value=event.target.value;$('frameSlider').dispatchEvent(new Event('input',{bubbles:true}));};
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&expanded)setExpanded(false);});
  new ResizeObserver(()=>{if(active&&mode==='mesh')drawMeshes();}).observe($('outputPanel'));
  refreshAssets();
 }
 function shader(gl,type,source){
  const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
  if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));
  return shader;
 }
 function renderer(canvas){
  if(renderers.has(canvas))return renderers.get(canvas);
  const gl=canvas.getContext('webgl',{alpha:false,antialias:true});
  if(!gl)throw Error('This browser cannot create a WebGL hand view; use UV atlas');
  const program=gl.createProgram();
  gl.attachShader(program,shader(gl,gl.VERTEX_SHADER,`
   attribute vec3 aPosition;attribute vec3 aNormal;attribute vec2 aUv;
   uniform mat3 uRotation;uniform float uZoom;uniform float uAspect;
   varying vec2 vUv;varying vec3 vNormal;
   void main(){vec3 p=uRotation*aPosition;vNormal=uRotation*aNormal;vUv=aUv;
    gl_Position=vec4(p.x*uZoom/uAspect,p.y*uZoom,-p.z*.25,1.0);}`));
  gl.attachShader(program,shader(gl,gl.FRAGMENT_SHADER,`
   precision mediump float;varying vec2 vUv;varying vec3 vNormal;uniform sampler2D uTexture;
   void main(){vec2 pixel=vec2(vUv.x,1.0-vUv.y)*223.0+.5;
    vec4 heat=texture2D(uTexture,pixel/224.0);
    float light=.68+.32*abs(dot(normalize(vNormal),normalize(vec3(.2,.4,1.0))));
    vec3 base=vec3(.15,.19,.25);gl_FragColor=vec4(mix(base,heat.rgb,heat.a)*light,1.0);}`));
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
  const buffer=gl.createBuffer(),texture=gl.createTexture();gl.useProgram(program);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  for(const [name,size,offset] of [['aPosition',3,0],['aNormal',3,12],['aUv',2,24]]){
   const index=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(index);gl.vertexAttribPointer(index,size,gl.FLOAT,false,32,offset);
  }
  gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);
  for(const parameter of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,parameter,gl.NEAREST);
  for(const parameter of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,parameter,gl.CLAMP_TO_EDGE);
  gl.uniform1i(gl.getUniformLocation(program,'uTexture'),0);gl.enable(gl.DEPTH_TEST);
  const instance={gl,program,buffer,texture,geometryKey:null};renderers.set(canvas,instance);
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{hooks.pause();drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.012;pitch+=(e.clientY-drag[1])*.012;drag=[e.clientX,e.clientY];drawMeshes();});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{drag=null;});
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.45,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));drawMeshes();},{passive:false});
  return instance;
 }
 const sub=(a,b)=>a.map((x,i)=>x-b[i]);
 const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
 const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 const unit=a=>{const n=Math.hypot(...a);if(n<1e-10)throw Error('Recorded hand basis is degenerate');return a.map(x=>x/n);};
 function meshGeometry(mesh,annotation){
  const j=annotation.joints,y=unit(sub(j[9],j[0]));
  let x=sub(j[17],j[5]);if(annotation.hand_side==='left')x=x.map(v=>-v);
  x=unit(sub(x,y.map(v=>v*dot(x,y))));const z=unit(cross(x,y));
  const positions=annotation.vertices.map(v=>{const p=sub(v,j[0]);return [dot(p,x),dot(p,y),dot(p,z)];});
  const low=[0,1,2].map(k=>Math.min(...positions.map(p=>p[k]))),high=[0,1,2].map(k=>Math.max(...positions.map(p=>p[k])));
  const center=low.map((v,k)=>(v+high[k])/2),scale=1.65/Math.max(...high.map((v,k)=>v-low[k]));
  const points=positions.map(p=>sub(p,center).map(x=>x*scale)),normals=points.map(()=>[0,0,0]);
  for(const tri of mesh.faces){const n=cross(sub(points[tri[1]],points[tri[0]]),sub(points[tri[2]],points[tri[0]]));for(const v of tri)normals[v]=normals[v].map((a,k)=>a+n[k]);}
  const packed=[];mesh.faces.forEach((tri,i)=>tri.forEach((v,k)=>packed.push(...points[v],...unit(normals[v]),...mesh.uv[mesh.uv_faces[i][k]])));
  return new Float32Array(packed);
 }
 function drawMeshes(){
  if(!active||mode!=='mesh'||!pose?.available||!current)return;
  const mesh=assets?.sides?.[current.hand_side]?.mesh;if(!mesh)return;
  try{
   if(!geometryCache||geometryCache.pose!==pose||geometryCache.mesh!==mesh)
    geometryCache={pose,mesh,values:meshGeometry(mesh,pose)};
   const geometry=geometryCache.values,cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch);
   const rotation=new Float32Array([cy,sy*sx,-sy*cx,0,cx,sx,sy,-cy*sx,cy*cx]);
   for(const kind of kinds){
    const canvas=$('pressureMesh'+kind),r=renderer(canvas),gl=r.gl;
    const pixels=Math.min(1024,Math.max(256,Math.round(canvas.getBoundingClientRect().width*(window.devicePixelRatio||1))));
    if(canvas.width!==pixels||canvas.height!==pixels)canvas.width=canvas.height=pixels;
    gl.viewport(0,0,pixels,pixels);gl.clearColor(.035,.05,.075,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(r.program);
    gl.bindBuffer(gl.ARRAY_BUFFER,r.buffer);
    if(r.geometryKey!==geometry){gl.bufferData(gl.ARRAY_BUFFER,geometry,gl.DYNAMIC_DRAW);r.geometryKey=geometry;}
    const maximum=Number($(kind==='error'?'errorScale':'pressureScale').value),values=current[kind],colorKey=maximum+':'+gamma();
    if(r.textureData!==current||r.colorKey!==colorKey){
     const texture=new Uint8Array(224*224*4);
     for(let i=0;i<224*224;i++){
      const value=values?.[i]??null,c=color(value,maximum),alpha=value===null?255:Math.round(255*Math.min(1,Math.max(0,value/maximum))**gamma());
      texture.set([...c,alpha],i*4);
     }
     gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,r.texture);
     gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,224,224,0,gl.RGBA,gl.UNSIGNED_BYTE,texture);
     r.textureData=current;r.colorKey=colorKey;
    }
    gl.uniformMatrix3fv(gl.getUniformLocation(r.program,'uRotation'),false,rotation);
    gl.uniform1f(gl.getUniformLocation(r.program,'uZoom'),zoom);gl.uniform1f(gl.getUniformLocation(r.program,'uAspect'),1);
    gl.drawArrays(gl.TRIANGLES,0,geometry.length/8);
   }
  }catch(error){$('pressureSurfaceStatus').textContent=error.message+' · UV atlas shown';$('pressureMeshGrid').hidden=true;$('outputPanel').querySelector('.maps').hidden=false;}
 }
 return {init,setup,update,clear,atlas,redraw:drawMeshes,meshGeometry};
})();
