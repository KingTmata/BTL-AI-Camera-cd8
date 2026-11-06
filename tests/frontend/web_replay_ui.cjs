const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict'),{test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
function setup(){
 const nodes=new Map(),calls=[],timers=[],revoked=[],decodes=[],created=[];let serial=0,now=1000,clockCalls=0;
 const node=()=>({value:'',checked:false,hidden:false,dataset:{},style:{},children:[],textContent:'',get options(){return this.children;},append(...x){this.children.push(...x);},replaceChildren(...x){this.children=x;},addEventListener(){},click(){},removeAttribute(name){delete this[name];},pause(){},load(){},getContext(){return {clearRect(){},resetTransform(){}};},querySelector(){return node();},querySelectorAll(){return [];}});
 const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
 get('processingMode').value='replay';get('sampleFPS').value='5';get('task').value='pose';
 let responses=[];
 class Image{constructor(){this.naturalWidth=8;this.naturalHeight=8;}decode(){return new Promise(resolve=>decodes.push(resolve));}}
 let timerSerial=0;
 const context=vm.createContext({document:{getElementById:get,createElement:node},performance:{now:()=>now},Image,Response,Headers,Blob,AbortController,TextDecoder,Uint8Array,Float64Array,DataView,URLSearchParams,URL:{createObjectURL(blob){const url='blob:'+ ++serial;created.push({url,blob});return url;},revokeObjectURL(value){revoked.push(value);}},setTimeout(fn,delay){const id=++timerSerial;timers.push({id,fn,delay});return id;},clearTimeout(id){const index=timers.findIndex(t=>t.id===id);if(index>=0)timers.splice(index,1);},fetch:async(url,options={})=>{if(url==='/api/clock'){clockCalls++;return Response.json({clock_id:'fixture',monotonic_seconds:100+now/1000});}calls.push({url:String(url),signal:options.signal});if(!responses.length)throw Error('Unexpected HTTP');const next=responses.shift();return typeof next==='function'?next():next;}});
 vm.runInContext(html.split('<script>')[1].split('if(videoFrameAPI){')[0],context);
 const run=code=>vm.runInContext(code,context);
 run(`job={id:'one',state:'processing',processing_mode:'replay',replay_settings:{sample_fps:5,tracking:'off'}};config={};fitOverlay=()=>{};renderArtifacts=()=>{};renderAnnotations=()=>{};updateControls=()=>{};updateSaveState=()=>{};`);
 return {get,run,calls,timers,revoked,decodes,created,time(value){now=value;},clockCalls(){return clockCalls;},respond(...items){responses.push(...items);}};
}
const latest=(sequence=0,state='processing')=>Response.json({state,sequence,update:{sequence,source_frame_index:sequence*6,frame_index:sequence*6,timestamp_seconds:sequence/5,source_pts:sequence*200,time_base:'1/1000',width:8,height:8,segment_id:0,detections:[],tracks:[],timing:{source_to_publish_ms:30}}});
const jpeg=sequence=>new Response(new Uint8Array([255,216,255,217]),{headers:{'content-type':'image/jpeg','x-replay-sequence':String(sequence)}});
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function expirePoll(s){const deadline=s.timers.find(t=>t.delay>1000);assert.ok(deadline,'in-flight network/decode needs a bounded poll deadline');deadline.fn();}

test('hung latest fetch times out only its poll and recovers on a later sample',async()=>{
 const s=setup();s.respond(()=>new Promise(()=>{}));const loading=s.run('startLiveReplay(jobEpoch)');await tick();expirePoll(s);await loading;
 assert.equal(s.calls[0].signal.aborted,true);assert.equal(s.run('liveSession.controller.signal.aborted'),false);
 assert.equal(s.run('currentFrame'),null);assert.match(s.get('frameStatus').textContent,/thời gian|timeout/i);
 assert.ok(s.timers.some(t=>t.delay>=1000));
 s.respond(latest(1),jpeg(1));const recovery=s.run('pollLiveReplay(liveSession)');await tick();s.decodes.shift()();await recovery;
 assert.equal(s.get('liveImage').dataset.sequence,'1');
});

test('hung latest body is canceled without poisoning the next poll',async()=>{
 const s=setup();let cancelled=0;const body=new ReadableStream({cancel(){cancelled++;}});
 s.respond(new Response(body,{headers:{'content-type':'application/json'}}));const loading=s.run('startLiveReplay(jobEpoch)');await tick();expirePoll(s);await loading;
 assert.equal(cancelled,1);assert.equal(s.run('currentFrame'),null);assert.equal(s.run('liveSession.finished'),false);
});

test('hung JPEG body is canceled before allocating an image URL',async()=>{
 const s=setup();let cancelled=0;const body=new ReadableStream({cancel(){cancelled++;}});
 s.respond(latest(),new Response(body,{headers:{'content-type':'image/jpeg','x-replay-sequence':'0'}}));const loading=s.run('startLiveReplay(jobEpoch)');await tick();expirePoll(s);await loading;
 assert.equal(cancelled,1);assert.equal(s.created.length,0);assert.equal(s.get('liveImage').dataset.sequence,undefined);
});

test('decode timeout revokes pending image and its late completion cannot replace a recovered sample',async()=>{
 const s=setup();s.respond(latest(),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();expirePoll(s);await loading;
 assert.deepEqual(s.revoked,['blob:1']);assert.equal(s.run('liveSession.pendingURL'),null);
 s.respond(latest(1),jpeg(1));const recovery=s.run('pollLiveReplay(liveSession)');await tick();
 s.decodes.shift()();await tick();assert.equal(s.get('liveImage').dataset.sequence,undefined);
 s.decodes.shift()();await recovery;assert.equal(s.get('liveImage').dataset.sequence,'1');assert.equal(s.run('currentFrame.sequence'),1);
 assert.equal(s.run('liveSession.timing.lifetime.commits'),1);
});

test('old-job abort releases a hung latest promise without waiting for its response',async()=>{
 const s=setup();s.respond(()=>new Promise(()=>{}));const loading=s.run('startLiveReplay(jobEpoch)');await tick();
 s.run(`stopLiveReplay();jobEpoch++;job={id:'two',state:'queued'};`);
 const released=await Promise.race([loading.then(()=>true),new Promise(resolve=>setTimeout(()=>resolve(false),100))]);
 assert.equal(released,true);assert.equal(s.run('currentFrame'),null);assert.equal(s.timers.length,0);
});
test('replay commits image and source overlay only after matching image decodes, and duplicate latest skips JPEG',async()=>{
 const s=setup();s.respond(latest(),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();
 assert.equal(s.get('liveImage').dataset.sequence,undefined);assert.equal(s.run('currentFrame'),null);
 s.decodes.shift()();await loading;
 assert.equal(s.get('liveImage').dataset.sequence,'0');assert.equal(s.run('currentFrame.sequence'),0);
 assert.equal(s.get('video').hidden,true);assert.equal(s.get('reviewPanel').hidden,true);
 assert.equal(s.get('liveImage').hidden,false);assert.equal(s.run('reviewReady'),false);
 s.respond(latest());await s.run('pollLiveReplay(liveSession)');
 assert.equal(s.calls.filter(c=>c.url.includes('/live-frame')).length,1);
 assert.ok(s.timers.every(t=>t.delay>=200));
});
test('stale sequence409 waits for next poll and never attaches mismatched metadata',async()=>{
 const s=setup();s.respond(latest(),new Response('',{status:409}));await s.run('startLiveReplay(jobEpoch)');
 assert.equal(s.calls.length,2);assert.equal(s.run('currentFrame'),null);
 assert.equal(s.get('liveImage').dataset.sequence,undefined);assert.equal(s.timers.length,1);
 assert.equal(s.timers[0].delay,200);
});
test('canceling during decode revokes the pending URL and cannot draw an old job',async()=>{
 const s=setup();s.respond(latest(),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();
 s.run(`stopLiveReplay();jobEpoch++;job={id:'two',state:'queued'};`);s.decodes.shift()();await loading;
 assert.equal(s.get('liveImage').dataset.sequence,undefined);assert.equal(s.run('currentFrame'),null);
 assert.deepEqual(s.revoked,['blob:1']);
 assert.equal(s.timers.length,0);
});
test('mismatched JPEG sequence is rejected and a terminal empty cache does not hot-poll',async()=>{
 const s=setup();s.respond(latest(),jpeg(3));await s.run('startLiveReplay(jobEpoch)');
 assert.equal(s.get('liveImage').dataset.sequence,undefined);assert.equal(s.decodes.length,0);
 s.respond(Response.json({state:'complete',sequence:null,update:null}));await s.run('pollLiveReplay(liveSession)');
 const count=s.timers.length;assert.equal(s.run('liveSession.finished'),true);
 assert.equal(s.timers.length,count);
});
test('replay completion does not load full preview or fetch review/GT context',async()=>{
 const s=setup();s.run(`job.state='complete';job.artifacts=['updates.jsonl'];job.summary={web_presentation_latency_ms:null};startLiveReplay=()=>{};api=()=>{throw Error('Review API forbidden');};`);
 await s.run('loadComplete(jobEpoch)');assert.equal(s.get('video').src,undefined);
 assert.equal(s.get('reviewPanel').hidden,true);assert.equal(s.run('reviewReady'),false);
});
test('switching to replay chooses Pose and scopes AI sampling controls without triggering inference',()=>{
 const s=setup();s.get('task').value='detect';s.run(`modelsForTask=()=>{};processingModeChanged();`);
 assert.equal(s.get('task').value,'pose');assert.equal(s.get('task').disabled,true);
 assert.equal(s.get('sampleFPS').disabled,true,'active replay keeps its settings locked');
 s.run(`job=null;processingModeChanged();`);assert.equal(s.get('sampleFPS').disabled,false);
 s.get('processingMode').value='full';s.run('processingModeChanged()');
 assert.equal(s.get('task').disabled,false);assert.equal(s.get('sampleFPS').disabled,true);
 assert.equal(s.calls.length,0);
});
test('old HTTP requests are aborted and failed decoding backs off without publishing an overlay',async()=>{
 const s=setup();let resolve;s.respond(()=>new Promise(r=>{resolve=r;}));
 const loading=s.run('startLiveReplay(jobEpoch)');await tick();s.run('stopLiveReplay();jobEpoch++;');
 assert.equal(s.calls[0].signal.aborted,true);resolve(latest());await loading;
 assert.equal(s.calls.length,1);assert.equal(s.timers.length,0);
});
test('overlay provenance is published only when the visible replay image shares its sequence',async()=>{
 const s=setup();s.respond(latest(),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();s.decodes.shift()();await loading;
 s.run('drawOverlay()');assert.equal(s.get('overlay').dataset.sequence,'0');assert.equal(s.get('overlay').dataset.frameIndex,'0');
 s.get('liveImage').dataset.sequence='other';s.run('drawOverlay()');
 assert.equal(s.get('overlay').dataset.sequence,undefined);assert.equal(s.get('overlay').dataset.frameIndex,undefined);
});
test('full submission ignores an invalid disabled replay sampling value',async()=>{
 const s=setup();s.run(`job=null;$('jobForm').reportValidity=()=>true;$('file').files=[];$('source').value='clip.mp4';$('model').value='pose';$('processingMode').value='full';$('sampleFPS').value='0';$('confidence').value='.05';$('imgsz').value='1280';$('device').value='cpu';$('tracking').value='off';$('nms').value='';let submitted;post=async(path,payload)=>{submitted=payload;return {id:'full',processing_mode:'full'};};poll=()=>{};clearViewer=()=>{};renderJob=()=>{};renderAnnotations=()=>{};`);
 await s.run('startJob({preventDefault(){}})');assert.equal(s.run('submitted.processing_mode'),'full');assert.equal(s.run('submitted.sample_fps'),5);
});
test('actual image commit records bounded client latency, exposes proof data, and downloads only local JSON',async()=>{
 const s=setup(),payload=await latest().json();payload.clock_id='fixture';payload.update.source_release_monotonic=100.9;payload.update.publish_monotonic=100.95;
 s.run(`replayClickStart={id:'one',client_ms:100};`);s.respond(Response.json(payload),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();
 assert.equal(s.clockCalls(),3);assert.equal(s.get('liveImage').dataset.replayTiming,undefined);
 s.time(1200);s.decodes.shift()();await loading;
 const proof=JSON.parse(s.get('liveImage').dataset.replayTiming);
 assert.equal(proof.lifetime.commits,1);assert.equal(proof.lifetime.source_to_commit.valid_count,1);
 assert.ok(Math.abs(proof.lifetime.source_to_commit.mean_upper_ms-300)<1e-7);
 assert.equal(proof.start_click_to_first_commit_ms,1100);assert.equal(proof.browser_paint_latency_ms,null);
 const calls=s.calls.length;s.run('downloadReplayTiming()');assert.equal(s.calls.length,calls,'download must not POST or send timing elsewhere');
 const exported=JSON.parse(await s.created.at(-1).blob.text());assert.equal(exported.records[0].source_pts,0);assert.equal(exported.records.length,1);
 assert.match(s.get('liveTimingStatus').textContent,/giải mã.*commit/);
});
test('clock calibration refreshes serially after60seconds and image polling remains bounded',async()=>{
 const s=setup();s.respond(latest(),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();s.decodes.shift()();await loading;
 assert.equal(s.clockCalls(),3);s.time(61001);s.respond(latest());await s.run('pollLiveReplay(liveSession)');
 assert.equal(s.clockCalls(),6);assert.equal(s.calls.filter(c=>c.url.includes('/live-frame')).length,1);
 assert.equal(s.timers.length,1);assert.equal(s.timers[0].delay,200);
});
const actionEnvelope=(ordinal,extra={})=>({event_id:'replay_run:'+ordinal,event:{label:'raise_hand',track_id:1,start_s:0,end_s:1,origin:'rule',review:'draft'},availability:{source_to_available_ms:17},...extra});
const allText=node=>[node.textContent,...node.children.map(allText)].join(' ');
test('optional replay rules default off, require explicit ByteTrack selection and remain disabled for full or busy jobs',()=>{
 const s=setup();assert.match(html,/<select id="actionRules"[^>]*><option value="off"/);
 s.run(`job=null;$('processingMode').value='full';updateProcessingOptions();`);assert.equal(s.get('actionRules').disabled,true);
 s.get('processingMode').value='replay';s.get('actionRules').value='raise_hand';s.get('tracking').value='off';s.run('actionRulesChanged()');
 assert.equal(s.get('tracking').value,'bytetrack');assert.equal(s.get('tracking').disabled,true);assert.equal(s.get('actionRules').disabled,false);
 assert.match(s.get('actionRulesNote').textContent,/ByteTrack/);assert.equal(s.calls.length,0);
 s.get('actionRules').value='off';s.run('actionRulesChanged()');assert.equal(s.get('tracking').disabled,false);
 s.run(`job={state:'processing'};updateProcessingOptions();`);assert.equal(s.get('actionRules').disabled,true);assert.equal(s.get('tracking').disabled,true);
});
test('submission sends optional rule only for replay and forces full mode off',async()=>{
 const s=setup();s.run(`job=null;$('jobForm').reportValidity=()=>true;$('file').files=[];$('source').value='clip.mp4';$('model').value='pose';$('confidence').value='.05';$('imgsz').value='1280';$('device').value='cpu';$('tracking').value='bytetrack';$('nms').value='';$('actionRules').value='raise_hand';let submitted=[];post=async(path,payload)=>{submitted.push(payload);return {id:'new',processing_mode:payload.processing_mode};};poll=()=>{};clearViewer=()=>{};renderJob=()=>{};renderAnnotations=()=>{};`);
 await s.run('startJob({preventDefault(){}})');assert.equal(s.run('submitted[0].action_rules'),'raise_hand');assert.equal(s.run('submitted[0].tracking'),'bytetrack');
 s.run(`job=null;$('processingMode').value='full';`);await s.run('startJob({preventDefault(){}})');assert.equal(s.run('submitted[1].action_rules'),'off');
});
test('draft rule counters and episodes attach only after the matching image commit, without touching human review',async()=>{
 const s=setup(),payload=await latest().json();payload.update.action_state={active_actor_count:0,pending_actor_count:2};payload.update.action_events=[actionEnvelope(0)];
 s.run(`job.replay_settings.action_rules='raise_hand';events=[{label:'HUMAN_UNSAVED'}];coverage=[{label:'HUMAN_COVERAGE'}];`);s.respond(Response.json(payload),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();
 assert.equal(s.get('liveActionEvents').children.length,0);s.decodes.shift()();await loading;
 assert.equal(s.get('liveActionPanel').hidden,false);assert.equal(s.get('liveActionEvents').children.length,1);
 assert.match(s.get('liveActionState').textContent,/Chưa nhận gợi ý/);assert.match(s.get('liveActionState').textContent,/2/);assert.match(s.get('liveActionState').textContent,/mẫu 0/i);
 assert.match(allText(s.get('liveActionEvents')),/Giơ tay/);assert.match(s.get('liveActionScope').textContent,/tập con.*đã nhận/);assert.match(s.get('liveActionScope').textContent,/không.*GT/i);
 assert.doesNotMatch(allText(s.get('liveActionPanel')),/17\s*ms/);assert.equal(s.run('events[0].label'),'HUMAN_UNSAVED');assert.equal(s.run('coverage[0].label'),'HUMAN_COVERAGE');
 s.get('actionRules').value='off';s.run('setReplayViewer()');assert.equal(s.get('liveActionPanel').hidden,false,'current job settings own visibility, not next-run input');
});
test('received episode list caps at twenty and ordinal watermark rejects duplicates, older and foreign IDs without growing memory',async()=>{
 const s=setup(),payload=await latest().json();payload.update.action_events=[actionEnvelope(0)];s.run(`job.replay_settings.action_rules='raise_hand';`);s.respond(Response.json(payload),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();s.decodes.shift()();await loading;
 s.run(`for(let i=1;i<=25;i++){const update={...currentFrame,sequence:i,action_state:{active_actor_count:1,pending_actor_count:0},action_events:[{event_id:'replay_run:'+i,event:{label:'raise_hand',track_id:1,start_s:0,end_s:1,origin:'rule',review:'draft'}}]};liveSession.lastSequence=i;currentFrame=update;$('liveImage').dataset.sequence=String(i);receiveReplayActions(liveSession,update);}`);
 assert.equal(s.get('liveActionEvents').children.length,20);assert.equal(s.run('liveSession.actionEpisodes.length'),20);assert.equal(s.run('liveSession.actionReceivedCount'),26);
 s.run(`receiveReplayActions(liveSession,{...currentFrame,action_events:[${JSON.stringify(actionEnvelope(0))},${JSON.stringify(actionEnvelope(25))},${JSON.stringify(actionEnvelope(50,{event_id:'foreign:50'}))}]});`);
 assert.equal(s.run('liveSession.actionReceivedCount'),26);assert.equal(s.run('liveSession.actionLastOrdinal'),25);
 assert.match(s.get('liveActionScope').textContent,/20/);assert.match(s.get('liveActionScope').textContent,/26/);
});
test('invalid event envelopes cannot poison watermark or become reviewed labels, and missing state stays unknown',async()=>{
 const s=setup(),payload=await latest().json();payload.update.action_events=[actionEnvelope(9,{event:{label:'write_board',track_id:1,start_s:0,end_s:1,origin:'rule',review:'draft'}}),actionEnvelope(10,{event_id:'bad:NaN'}),actionEnvelope(11,{event:{label:'raise_hand',track_id:1,start_s:0,end_s:1,origin:'human',review:'reviewed'}}),actionEnvelope(0)];s.run(`job.replay_settings.action_rules='raise_hand';`);s.respond(Response.json(payload),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();s.decodes.shift()();await loading;
 assert.equal(s.run('liveSession.actionReceivedCount'),1);assert.equal(s.run('liveSession.actionLastOrdinal'),0);assert.equal(s.run('liveSession.actionPrefix'),'replay_run');assert.match(s.get('liveActionState').textContent,/Chưa nhận trạng thái/);
});
test('abort clears draft panel and an old decode cannot attach episodes to another job',async()=>{
 const s=setup(),payload=await latest().json();payload.update.action_events=[actionEnvelope(0)];s.run(`job.replay_settings.action_rules='raise_hand';`);s.respond(Response.json(payload),jpeg(0));const loading=s.run('startLiveReplay(jobEpoch)');await tick();s.run(`stopLiveReplay();jobEpoch++;job={id:'two',state:'queued',processing_mode:'replay',replay_settings:{action_rules:'off'}};setReplayViewer();`);s.decodes.shift()();await loading;
 assert.equal(s.get('liveActionPanel').hidden,true);assert.equal(s.get('liveActionEvents').children.length,0);assert.equal(s.get('liveImage').dataset.sequence,undefined);
});
test('terminal rule report explains cadence and durable flush without inventing received episodes or accuracy',()=>{
 const s=setup();s.run(`job.state='complete';job.replay_settings.action_rules='raise_hand';job.artifacts=['actions.jsonl'];job.summary={action_rules:{limitations:['nominal_cadence_below_rule_max_gap']},action_summary:{emitted_draft_events:7,terminal_flush_events:2}};startLiveReplay=()=>{};loadReplayComplete(jobEpoch);`);
 assert.equal(s.get('liveActionPanel').hidden,false);assert.equal(s.get('liveActionEvents').children.length,0);assert.match(s.get('liveActionReport').textContent,/7/);assert.match(s.get('liveActionReport').textContent,/lấy mẫu.*khoảng|khoảng.*lấy mẫu/);assert.match(s.get('liveActionReport').textContent,/kết thúc.*ảnh|ảnh.*kết thúc/);
 assert.equal(s.get('liveActionsArtifact').href,'/api/jobs/one/artifacts/actions.jsonl');assert.equal(s.get('liveActionsArtifact').hidden,false);
 assert.doesNotMatch(s.get('liveActionReport').textContent,/nominal_cadence_below_rule_max_gap/);
});
const captionMetadata=()=>({schema_version:1,kind:'full_frame_action_labels',job_id:'one',source_sha256:'a'.repeat(64),actor_namespace:'run:full',generation:'b'.repeat(32),frame_count:96,raw_frames_sha256:'c'.repeat(64),ground_truth:false,action_rules:'basic_pose',board_roi:null,board_roi_source_sha256:null,rule_config:{}});
const captionRow=index=>({frame_index:index,source_pts:index*1000,time_base:'1/30000',timestamp_seconds:index/30,segment_id:0,action_people:[{segment_id:0,track_id:7,status:'unsupported',label:null,origin:'rule',review:'draft',calibrated:false,rule:'coco17_wrist_above_nose_v1',observed_through_s:index/30}]});
const captionBundle=start=>({...captionMetadata(),start,count:32,frames:Array.from({length:32},(_,i)=>captionRow(start+i))});
function fullCaptions(s){s.run(`job={id:'one',state:'complete',processing_mode:'full',preview_verified:true};source={frame_count:96};frameTimes=Float64Array.from({length:96},(_,i)=>i/30);reviewSourceHash='${'a'.repeat(64)}';actorNamespace='run:full';$('showActions').checked=true;drawOverlay=()=>{};let prepares=0,windows=[];api=async()=>{prepares++;return ${JSON.stringify(captionMetadata())};};readFrameBundle=async(path,signal)=>{windows.push({path,signal});const start=Number(new URLSearchParams(path.split('?')[1]).get('start'));return {data:{...${JSON.stringify(captionBundle(0))},start,frames:Array.from({length:32},(_,i)=>({...${JSON.stringify(captionRow(0))},frame_index:start+i,source_pts:(start+i)*1000,timestamp_seconds:(start+i)/30,action_people:[]}))},bytes:2000};};requestedFrame=0;`);}
test('full action captions prepare saved Pose once and keep only one bounded32-frame window without touching human annotations',async()=>{
 const s=setup();fullCaptions(s);s.run(`events=[{label:'HUMAN'}];coverage=[{label:'COVERAGE'}];`);await s.run('prepareActionCaptions()');await s.run('ensureActionCaptionWindow(0)');await s.run('prepareActionCaptions()');
 assert.equal(s.run('prepares'),1);assert.equal(s.run('windows.length'),1);assert.equal(s.run('actionCaptionSession.frames.size'),32);assert.match(s.run('windows[0].path'),/action-labels\?start=0&count=32/);
 s.run('requestedFrame=70;');await s.run('ensureActionCaptionWindow(70)');assert.equal(s.run('actionCaptionSession.frames.size'),32);assert.equal(s.run('actionCaptionSession.frames.has(0)'),false);assert.equal(s.run('actionCaptionSession.frames.has(70)'),true);
 assert.equal(s.run('events[0].label'),'HUMAN');assert.equal(s.run('coverage[0].label'),'COVERAGE');
});
test('seeking cancels the older action window and delayed metadata cannot repopulate cache or attach to a new job',async()=>{
 const s=setup();fullCaptions(s);await s.run('prepareActionCaptions()');s.run(`let pending=[];readFrameBundle=(path,signal)=>new Promise(resolve=>pending.push({path,signal,resolve}));requestedFrame=32;`);
 const old=s.run('ensureActionCaptionWindow(32)');s.run('requestedFrame=70;');const current=s.run('ensureActionCaptionWindow(70)');assert.equal(s.run('pending[0].signal.aborted'),true);
 s.run(`pending[1].resolve({data:${JSON.stringify(captionBundle(64))},bytes:2000});`);await current;s.run(`pending[0].resolve({data:${JSON.stringify(captionBundle(32))},bytes:2000});`);await old;
 assert.equal(s.run('actionCaptionSession.frames.has(70)'),true);assert.equal(s.run('actionCaptionSession.frames.has(32)'),false);
 s.run('requestedFrame=0;');const other=s.run('ensureActionCaptionWindow(0)');s.run(`resetActionCaptions();jobEpoch++;job={id:'two'};pending[2].resolve({data:${JSON.stringify(captionBundle(0))},bytes:2000});`);await other;assert.equal(s.run('actionCaptionSession'),null);
});
test('full caption generation/source/namespace/timestamp mismatch is rejected once rather than looping per frame',async()=>{
 for(const fault of [{generation:'d'.repeat(32)},{source_sha256:'d'.repeat(64)},{actor_namespace:'run:wrong'},{frames:[{...captionRow(0),timestamp_seconds:9},...captionBundle(0).frames.slice(1)]}]){
  const s=setup();fullCaptions(s);s.run(`readFrameBundle=async()=>({data:{...${JSON.stringify(captionBundle(0))},...${JSON.stringify(fault)}},bytes:2000});`);await s.run('prepareActionCaptions()');await s.run('ensureActionCaptionWindow(0)');
  assert.equal(s.run('actionCaptionSession.frames.size'),0);assert.equal(s.run('actionCaptionSession.failed'),true);await s.run('ensureActionCaptionWindow(32)');assert.equal(s.run('prepares'),1);assert.match(s.get('actionCaptionStatus').textContent,/khớp|hợp lệ/);
 }
});
test('unsupported full jobs report one preparation failure and toggle-off during prepare prevents rendering or window fetches',async()=>{
 const failed=setup();fullCaptions(failed);failed.run(`api=async()=>{prepares++;const e=Error('Cần Pose và ByteTrack');e.status=409;throw e;};`);await failed.run('prepareActionCaptions()');await failed.run('prepareActionCaptions()');await failed.run('ensureActionCaptionWindow(0)');assert.equal(failed.run('prepares'),1);assert.match(failed.get('actionCaptionStatus').textContent,/Pose và ByteTrack/);
 const s=setup();fullCaptions(s);s.run(`let finishPrepare;api=()=>new Promise(resolve=>finishPrepare=resolve);`);const preparing=s.run('prepareActionCaptions()');s.get('showActions').checked=false;s.run(`showActionsChanged();finishPrepare(${JSON.stringify(captionMetadata())});`);await preparing;
 assert.equal(s.run('windows.length'),0);assert.equal(s.run('actionCaptionSession.prepared'),true);s.get('showActions').checked=true;await s.run('showActionsChanged()');assert.equal(s.run('windows.length'),1);
});
test('native frame/PTS/timebase/scene mismatch cannot attach cached caption to the current displayed person',async()=>{
 const s=setup();fullCaptions(s);s.run(`readFrameBundle=async()=>({data:${JSON.stringify(captionBundle(0))},bytes:2000});`);await s.run('prepareActionCaptions()');s.run(`currentFrame=${JSON.stringify(captionRow(0))};currentFrameIndex=0;`);assert.equal(s.run('actionCaptionFrame().frame_index'),0);
 for(const [key,value] of [['source_pts',1],['time_base','1/1000'],['segment_id',3],['timestamp_seconds',.1]]){s.run(`currentFrame={...${JSON.stringify(captionRow(0))},${key}:${JSON.stringify(value)}};`);assert.equal(s.run('actionCaptionFrame()'),null,key);}
});
test('verified full load automatically prepares requested captions and displayed cached frames request the matching action window',async()=>{
 const s=setup();fullCaptions(s);s.run(`let automatic=0,requested=[];prepareActionCaptions=()=>{automatic++;return Promise.resolve();};ensureActionCaptionWindow=index=>{requested.push(index);return Promise.resolve();};loadFrameTimes=async()=>{frameTimes=Float64Array.from({length:96},(_,i)=>i/30);return true;};primeFrames=async()=>{};readClassReviewContext=async()=>({source_sha256:'${'a'.repeat(64)}',actor_namespace:'run:full',duration_s:3.2,events:[],coverage:[],track_segments:[],review_metadata:{}});renderActors=()=>{};renderActorReview=()=>{};`);
 await s.run('loadComplete(jobEpoch)');assert.equal(s.run('automatic'),1);
 s.run(`requestedFrame=18;frameCache.set(18,${JSON.stringify(captionRow(18))});showCachedFrame();`);assert.equal(s.run('requested.at(-1)'),18);
 s.run(`let resetCount=0;resetActionCaptions=()=>resetCount++;clearViewer();`);assert.equal(s.run('resetCount'),1);
});
test('board region uses normalized source-canvas corners, persists only by verified source hash and rebuilds full hints without changing GT',async()=>{
 const s=setup();fullCaptions(s);s.run(`let saved=[];storageSet=(key,value)=>saved.push({key,value});storageGet=()=>null;loadBoardROI('${'a'.repeat(64)}');events=[{label:'HUMAN'}];coverage=[];let rebuilt=[];prepareActionCaptions=()=>{rebuilt.push(boardROI);return Promise.resolve();};$('overlay').getBoundingClientRect=()=>({left:100,top:200,width:800,height:450});startBoardSelection();`);
 s.run(`boardPointerDown({clientX:180,clientY:470,preventDefault(){}});boardPointerUp({clientX:180,clientY:470,preventDefault(){}});boardPointerDown({clientX:740,clientY:605,preventDefault(){}});boardPointerUp({clientX:740,clientY:605,preventDefault(){}});`);
 assert.deepEqual(JSON.parse(s.run('JSON.stringify(boardROI)')),[.1,.6,.8,.9]);assert.equal(s.run('boardDrawing'),false);assert.equal(s.get('overlay').style.pointerEvents,'none');
 assert.equal(s.run('saved[0].key'),'classroom-demo-board-roi-'+ 'a'.repeat(64));assert.equal(s.run('rebuilt.length'),1);assert.equal(s.run('events[0].label'),'HUMAN');assert.equal(s.run('coverage.length'),0);
 s.run(`storageGet=()=>JSON.stringify([0,0,0,1]);loadBoardROI('${'d'.repeat(64)}');`);assert.equal(s.run('boardROI'),null);
});
test('full basic preparation pins ROI hash/config and GET generation while ROI changes abort prior sidecar responses',async()=>{
 const s=setup();fullCaptions(s);const roi=[.1,.6,.8,.9];s.run(`loadBoardROI('${'a'.repeat(64)}');boardROI=${JSON.stringify(roi)};let bodies=[];api=async(path,options)=>{bodies.push(JSON.parse(options.body));return {...${JSON.stringify(captionMetadata())},board_roi:${JSON.stringify(roi)},board_roi_source_sha256:'${'a'.repeat(64)}'};};readFrameBundle=async(path)=>{windows.push({path});return {data:{...${JSON.stringify(captionBundle(0))},board_roi:${JSON.stringify(roi)},board_roi_source_sha256:'${'a'.repeat(64)}'},bytes:2000};};`);
 await s.run('prepareActionCaptions()');assert.equal(s.run('bodies[0].action_rules'),'basic_pose');assert.deepEqual(JSON.parse(s.run('JSON.stringify(bodies[0].board_roi)')),roi);assert.equal(s.run('bodies[0].board_roi_source_sha256'),'a'.repeat(64));assert.match(s.run('windows[0].path'),/generation=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/);
 s.run(`let oldSession=actionCaptionSession;prepareActionCaptions=()=>Promise.resolve();`);await s.run('setBoardROI(null)');assert.equal(s.run('oldSession.controller.signal.aborted'),true);assert.equal(s.run('actionCaptionSession'),null);
});
test('replay basic sends ROI for the verified source only and form edits never alter current replay configuration',async()=>{
 const s=setup();s.run(`job=null;$('processingMode').value='replay';$('actionRules').value='basic_pose';$('tracking').value='off';actionRulesChanged();`);assert.equal(s.get('tracking').value,'bytetrack');
 s.run(`job={id:'one',state:'processing',processing_mode:'replay',replay_settings:{action_rules:'basic_pose',board_roi:[.2,.3,.5,.7]}};boardROISourceHash='${'a'.repeat(64)}';boardROI=[.1,.6,.8,.9];boardROIReplayEligible=true;events=[{label:'HUMAN'}];`);
 await s.run('setBoardROI([.1,.5,.7,.8])');assert.deepEqual(JSON.parse(s.run('JSON.stringify(job.replay_settings.board_roi)')),[.2,.3,.5,.7]);assert.match(s.get('boardROIStatus').textContent,/lượt.*sau|lượt kế tiếp/);
 s.run(`invalidateReplayBoardROI();`);assert.equal(s.run('boardROIReplayEligible'),false);
});
test('accessible numeric ROI fields apply exact normalized bounds and reject empty or inverted entries',async()=>{
 const s=setup();fullCaptions(s);s.run(`storageGet=()=>null;loadBoardROI('${'a'.repeat(64)}');prepareActionCaptions=()=>Promise.resolve();$('boardX1').value='.48';$('boardY1').value='.40';$('boardX2').value='.75';$('boardY2').value='.68';`);
 await s.run('applyBoardROIFields()');assert.deepEqual(JSON.parse(s.run('JSON.stringify(boardROI)')),[.48,.4,.75,.68]);
 s.get('boardX1').value='';await s.run('applyBoardROIFields()');assert.deepEqual(JSON.parse(s.run('JSON.stringify(boardROI)')),[.48,.4,.75,.68]);assert.match(s.get('errorText').textContent,/đủ|tọa độ/);
 s.get('boardX1').value='.9';await s.run('applyBoardROIFields()');assert.deepEqual(JSON.parse(s.run('JSON.stringify(boardROI)')),[.48,.4,.75,.68]);assert.match(html,/id="applyBoardROI"/);
});
test('ROI change removes old-generation captions before redrawing its new rectangle',async()=>{
 const s=setup();fullCaptions(s);await s.run('prepareActionCaptions()');s.run(`loadBoardROI('${'a'.repeat(64)}');let staleDraws=0;drawOverlay=()=>{if(actionCaptionSession?.frames.size)staleDraws++;};prepareActionCaptions=()=>Promise.resolve();`);
 await s.run('setBoardROI([.1,.2,.6,.7])');assert.equal(s.run('staleDraws'),0);
});
test('replay ROI submission is basic-only and source changes omit both ROI and its binding',()=>{
 const s=setup();s.run(`boardROI=[.1,.2,.6,.7];boardROISourceHash='${'a'.repeat(64)}';boardROIReplayEligible=true;$('actionRules').value='basic_pose';`);assert.deepEqual(JSON.parse(s.run('JSON.stringify(replayBoardROIInput().board_roi)')),[.1,.2,.6,.7]);
 for(const mode of ['raise_hand','off']){s.get('actionRules').value=mode;assert.equal(s.run('replayBoardROIInput().board_roi'),null);}
 s.get('actionRules').value='basic_pose';s.run('invalidateReplayBoardROI()');assert.equal(s.run('replayBoardROIInput().board_roi'),null);assert.equal(s.run('replayBoardROIInput().board_roi_source_sha256'),null);
});
test('Full sidecar snapshots hand cue, validates returned cue and rebuilds on selection changes',async()=>{
 const s=setup();fullCaptions(s);s.get('handCue').value='shoulder_any_arm_v1';
 s.run(`let bodies=[];api=async(_path,options)=>{bodies.push(JSON.parse(options.body));return {...${JSON.stringify(captionMetadata())},hand_cue:'shoulder_any_arm_v1'};};readFrameBundle=async()=>({data:{...${JSON.stringify(captionBundle(0))},hand_cue:'shoulder_any_arm_v1'},bytes:2000});`);
 await s.run('prepareActionCaptions()');assert.equal(s.run('bodies[0].hand_cue'),'shoulder_any_arm_v1');assert.equal(s.run('actionCaptionSession.handCue'),'shoulder_any_arm_v1');assert.equal(s.run('actionCaptionSession.prepared'),true);
 s.run(`let oldCueSession=actionCaptionSession;ensureActionCaptionWindow=()=>Promise.resolve();api=async(_path,options)=>{bodies.push(JSON.parse(options.body));return {...${JSON.stringify(captionMetadata())},hand_cue:'nose_both_arms_v1'};};`);
 s.get('handCue').value='nose_both_arms_v1';await s.run('handCueChanged()');assert.equal(s.run('oldCueSession.controller.signal.aborted'),true);assert.equal(s.run('bodies[1].hand_cue'),'nose_both_arms_v1');
});
test('a shoulder-cue request rejects legacy or malformed sidecar metadata',async()=>{
 for(const cue of [undefined,'invented','nose_both_arms_v1']){
  const s=setup();fullCaptions(s);s.get('handCue').value='shoulder_any_arm_v1';const metadata={...captionMetadata()};if(cue!==undefined)metadata.hand_cue=cue;
  s.run(`api=async()=>(${JSON.stringify(metadata)});`);await s.run('prepareActionCaptions()');assert.equal(s.run('actionCaptionSession.failed'),true,String(cue));assert.equal(s.run('windows.length'),0);
 }
});
