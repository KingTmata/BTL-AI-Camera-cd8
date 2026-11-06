// Real inline script, delayed streaming HTTP responses, and a minimal canvas DOM.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const html = fs.readFileSync(require('node:path').join(__dirname,'../../src/classroom_ai/interfaces/web.html'), 'utf8');
const script = html.split('<script>')[1].split('if(videoFrameAPI){')[0];
const tick = () => new Promise(resolve => setTimeout(resolve, 5));
function setup(count=200) {
  const nodes = new Map(), calls = [], pending = [];
  const node = () => ({dataset:{},style:{},children:[],checked:false,width:640,height:480,readyState:0,paused:true,currentTime:0,
    getContext(){return {clearRect(){},setLineDash(){}};},requestVideoFrameCallback(callback){this.callback=callback;},pause(){},load(){},removeAttribute(name){delete this[name];},
    querySelector(){return {};},replaceChildren(){},append(){},addEventListener(){}});
  const get = id => {if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
  const fetch = async (url, options={}) => {
    calls.push(String(url));
    if(String(url).endsWith('/review-context')||String(url).includes('/class-summary?'))return Response.json({track_segments:[],events:[],coverage:[],class_summary:{session:{},windows:[],saved_intervals:{}}});
    return new Promise(resolve => pending.push({url:String(url),signal:options.signal,resolve}));
  };
  const context = vm.createContext({document:{getElementById:get,createElement:node},fetch,Response,Headers,AbortController,TextDecoder,Uint8Array,Float64Array,DataView,Number,JSON,Error,Date,console});
  vm.runInContext(script,context);
  vm.runInContext(html.slice(html.indexOf('if(videoFrameAPI){'),html.indexOf("$('dismissError').addEventListener")),context);
  const run = code => vm.runInContext(code,context);
  run(`job={id:'one',state:'complete',preview_verified:true,source:{frame_count:${count}},artifacts:['preview.mp4']};
    config={};frameTimes=Float64Array.from({length:${count}},(_,i)=>i/30);
    renderArtifacts=()=>{};renderActors=()=>{};renderAnnotations=()=>{};updateControls=()=>{};updateSaveState=()=>{};
    loadFrameTimes=async()=>{frameTimes=Float64Array.from({length:${count}},(_,i)=>i/30);return true;};`);
  function release(request, padding=0) {
    const url = new URL(request.url,'http://local');
    if(!url.pathname.endsWith('/frames')) {request.resolve(Response.json({detail:'per-frame HTTP forbidden'},{status:500}));return;}
    const start = Number(url.searchParams.get('start')), length = Math.min(Number(url.searchParams.get('count')),count-start);
    const frames = Array.from({length},(_,i)=>({frame_index:start+i,timestamp_seconds:(start+i)/30,detections:[],padding:'x'.repeat(padding)}));
    request.resolve(Response.json({start,count:length,frames}));
  }
  function releaseCapped(request,rowFactory) {
    const url=new URL(request.url,'http://local'),start=Number(url.searchParams.get('start'));
    const length=Math.min(Number(url.searchParams.get('count')),count-start);
    // Mirror the endpoint's byte budget, stopping before constructing a huge range.
    const frames=[];let bytes=Buffer.byteLength(JSON.stringify({start,count:length,frames}));
    for(let i=0;i<length;i++){
      const row=rowFactory(start+i);bytes+=Buffer.byteLength(JSON.stringify(row))+(i?1:0);
      if(bytes>4*1024*1024){request.resolve(Response.json({detail:'reduce count'},{status:413}));return bytes;}
      frames.push(row);
    }
    const body=JSON.stringify({start,count:length,frames});
    assert.equal(Buffer.byteLength(body),bytes);
    request.resolve(new Response(body,{headers:{'Content-Type':'application/json'}}));
    return bytes;
  }
  return {run,get,calls,pending,release,releaseCapped};
}
function poseRow(index) {
  // Synthetic 23-person frame with the production raw/track/display pose shape.
  // Full precision coordinates and both 17- and 13-joint arrays are retained.
  const people=Array.from({length:23},(_,id)=>{
    const keypoints17=Array.from({length:17},(_,i)=>[1047.8631591796875+i/97,408.5848693847656+i/101,0.42676296830177307+i/107]);
    return {class_id:0,class_name:'person',model_class_id:0,confidence:0.42676296830177307,bbox_xyxy:[1047.8631591796875,408.5848693847656,1130.1629638671875,491.0094299316406],keypoints17,keypoints13:keypoints17.slice(0,13),keypoints13_missing:[],visibility:0.9932477474212646,track_id:id,confirmed:true,detection_index:id,state:'observed'};
  });
  return {frame_index:index,timestamp_seconds:index/30,detections:people,tracks:people,display_tracks:people};
}
async function answerNew(s,factory,handled=new Set()) {
  for(const request of s.pending)if(!handled.has(request)){handled.add(request);s.releaseCapped(request,factory);}
  await tick();return handled;
}
test('verified load primes two blocks before exposing native playback; all 60 callbacks hit cache',async()=>{
  const s=setup(60), loading=s.run('loadComplete(jobEpoch)');
  await tick();
  assert.equal(s.get('video').src,undefined,'native controls must not play before metadata is primed');
  assert.equal(s.pending.length,2);
  s.pending.forEach(p=>s.release(p));await loading;
  assert.equal(s.get('video').src,'/api/jobs/one/artifacts/preview.mp4');
  for(let i=0;i<60;i++) {
    s.get('video').callback(0,{mediaTime:i/30});
    assert.equal(s.get('overlay').dataset.frameIndex,String(i));
    assert.equal(Number(s.get('overlay').dataset.mediaTime),i/30);
  }
  assert.equal(s.calls.filter(u=>u.includes('/frames?')).length,2);
  assert.equal(s.calls.some(u=>/\/frame\//.test(u)),false);
});
test('seek clears immediately, single-flights its window, and delayed old windows cannot replace needed rows',async()=>{
  const s=setup();s.run('requestFrame(0)');await tick();const old=s.pending.splice(0);
  s.run('requestFrame(4);requestFrame(4);requestFrame(4.033333333333333)');await tick();
  assert.equal(s.get('overlay').dataset.frameIndex,undefined);
  assert.equal(s.pending.length,2,'only one HTTP request per desired block');
  s.pending.forEach(p=>s.release(p));await tick();
  assert.equal(s.get('overlay').dataset.frameIndex,'121');
  old.reverse().forEach(p=>s.release(p));await tick();
  assert.equal(s.get('overlay').dataset.frameIndex,'121');
  assert.equal(s.run('frameCache.has(0)'),false,'obsolete response must not enter near-seek cache');
  assert.equal(s.run('frameCache.has(121)'),true);
});
test('old-job responses cannot repopulate a new job after clearViewer',async()=>{
  const s=setup();s.run('requestFrame(0)');await tick();const old=s.pending.splice(0);
  s.run("jobEpoch++;clearViewer();job={id:'two',state:'complete',preview_verified:true};frameTimes=Float64Array.from({length:200},(_,i)=>i/30);requestFrame(4);");
  await tick();s.pending.forEach(p=>s.release(p));await tick();
  old.forEach(p=>s.release(p));await tick();
  assert.equal(s.get('overlay').dataset.frameIndex,'120');
  assert.equal(s.run('frameCache.has(0)'),false);
});
test('sequential playback and seeks bound cache entries and serialized metadata bytes',async()=>{
  const s=setup(1024);let released=0;
  for(let index=0;index<1024;index+=32){
    s.run(`requestFrame(${index}/30)`);await tick();
    while(released<s.pending.length)s.release(s.pending[released++],100000);
    await tick();
    assert.equal(s.get('overlay').dataset.frameIndex,String(index));
    assert.ok(s.run('frameCache.size')<=96);
    assert.ok(s.run('frameCacheBytes')<=8*1024*1024);
  }
});
test('oversized streamed metadata and timestamp mismatches never enter the cache',async()=>{
  const s=setup();s.run('requestFrame(0)');await tick();
  const encoder=new TextEncoder();
  s.pending[0].resolve(new Response(new ReadableStream({start(controller){controller.enqueue(encoder.encode('x'.repeat(4*1024*1024+1)));controller.close();}})));
  s.pending[1].resolve(Response.json({start:32,count:32,frames:Array.from({length:32},(_,i)=>({frame_index:32+i,timestamp_seconds:-1}))}));
  await tick();
  assert.equal(s.run('frameCache.size'),0);
  assert.equal(s.run('frameCacheBytes'),0);
  assert.equal(s.get('overlay').dataset.frameIndex,undefined);
});
test('dense classroom ranges learn 16 rows, prime before native playback, and retain that size until a new job',async()=>{
  const s=setup(128),loading=s.run('loadComplete(jobEpoch)');await tick();
  const [first,oldNext]=s.pending.splice(0);
  assert.ok(s.releaseCapped(first,poseRow)>4*1024*1024);
  await tick();assert.equal(s.pending.length,2);
  assert.ok(s.pending.every(p=>new URL(p.url,'http://local').searchParams.get('count')==='16'));
  for(const p of s.pending)assert.ok(s.releaseCapped(p,poseRow)<4*1024*1024);
  await loading;
  assert.equal(s.get('video').src,'/api/jobs/one/artifacts/preview.mp4');
  for(let i=0;i<32;i++){s.get('video').callback(0,{mediaTime:i/30});assert.equal(s.get('overlay').dataset.frameIndex,String(i));}
  s.releaseCapped(oldNext,poseRow);await tick();
  assert.equal(s.run('frameBlockSize'),16);
  assert.equal(s.run('frameCache.has(0)'),true);
  assert.ok(s.run('frameCacheBytes')<=8*1024*1024);
  s.run("jobEpoch++;clearViewer();job={id:'two',state:'complete',preview_verified:true};frameTimes=Float64Array.from({length:128},(_,i)=>i/30);requestFrame(0);");
  assert.ok(s.pending.slice(-2).every(p=>new URL(p.url,'http://local').searchParams.get('count')==='32'));
});
test('a late successful response from the former block size cannot repopulate the resized cache',async()=>{
  const s=setup(128);s.run('requestFrame(0)');await tick();
  const [oldCurrent,oldNext]=s.pending.splice(0);
  s.releaseCapped(oldNext,poseRow);await tick();
  s.pending.forEach(p=>s.release(p));await tick();
  s.release(oldCurrent);await tick();
  assert.equal(s.run('frameBlocks.get(0).indices.length'),16);
  assert.equal(s.run('frameBlockSize'),16);
  assert.equal(s.run('frameCache.size'),32);
});
test('halving reaches one row; a permanently oversized row is remembered without a callback retry loop',async()=>{
  const s=setup(128),handled=new Set();s.run('requestFrame(0)');
  const huge=index=>({frame_index:index,timestamp_seconds:index/30,detections:[],padding:'x'.repeat(4*1024*1024+1)});
  for(let i=0;i<8;i++)await answerNew(s,huge,handled);
  assert.equal(s.run('frameFlights.size'),0);
  const calls=s.calls.length;
  for(let i=0;i<100;i++)s.get('video').callback(0,{mediaTime:0});
  await tick();assert.equal(s.calls.length,calls);
  assert.equal(s.run('frameBlockSize'),1);
  assert.match(s.get('frameStatus').textContent,/4 MiB/);
  assert.ok(calls<=12,'at most two requests per halving level');
  s.run('requestFrame(4)');await tick();s.pending.filter(p=>!handled.has(p)).forEach(p=>s.release(p));await tick();
  assert.equal(s.get('overlay').dataset.frameIndex,'120','other seek windows remain usable');
});
test('recoverable large rows learn one-row windows and evict before exceeding the byte budget',async()=>{
  const s=setup(128),handled=new Set(),loading=s.run('loadComplete(jobEpoch)');
  const large=index=>({frame_index:index,timestamp_seconds:index/30,detections:[],padding:'x'.repeat(3*1024*1024)});
  for(let i=0;i<8;i++)await answerNew(s,large,handled);
  await loading;
  assert.equal(s.run('frameBlockSize'),1);
  assert.equal(s.get('overlay').dataset.frameIndex,'0');
  for(let index=1;index<5;index++){
    s.get('video').callback(0,{mediaTime:index/30});
    assert.equal(s.get('overlay').dataset.frameIndex,String(index));
    await answerNew(s,large,handled);
    assert.ok(s.run('frameCacheBytes')<=8*1024*1024);
    assert.ok(s.run('frameCache.size')<=2);
  }
});
