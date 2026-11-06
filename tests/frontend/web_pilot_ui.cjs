const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict');
const {test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
const hash='a'.repeat(64),namespace='run:pilot';
function setup(){
 const nodes=new Map();
 const node=()=>({value:'',dataset:{},style:{},children:[],get options(){return this.children;},listeners:{},checked:false,hidden:false,textContent:'',append(...items){this.children.push(...items);},replaceChildren(...items){this.children=[...items];this.textContent='';},addEventListener(name,fn){this.listeners[name]=fn;},setAttribute(name,value){this[name]=value;},getContext(){return {clearRect(){}};},querySelector(){return node();},pause(){},load(){},removeAttribute(){}});
 const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
 const context=vm.createContext({document:{getElementById:get,createElement:node},Headers,URLSearchParams,AbortController,TextDecoder,Uint8Array,Float64Array,DataView,localStorage:{getItem(){return null;}}});
 vm.runInContext(html.split('<script>')[1].split('if(videoFrameAPI){')[0],context);
 const run=code=>vm.runInContext(code,context),text=n=>[n.textContent,...n.children.map(text)].join(' ');
 run(`job={id:'one',state:'complete',preview_verified:true};reviewReady=true;reviewSourceHash='${hash}';actorNamespace='${namespace}';activeReviewBranch='group-a';requestFrame=()=>{};`);
 return {get,run,text};
}
function fixture(){
 const windows=Array.from({length:6},(_,i)=>({window_id:'pilot-'+i,start_s:i*8.003333333333334,end_s:(i+1)*8.003333333333334,source_first_frame:i*240,source_last_frame:i*240+239,source_start_pts:i*4802,source_last_pts:i*4802+4782,time_base:'1/600'}));
 return {job_id:'one',review_branch:'group-a',source_sha256:hash,actor_namespace:namespace,review_snapshot:null,pilot:{pilot_id:'thanh',source_duration_seconds:48.02,approved_source_split:'train',windows},audit:{scope:'declared_human_actors',saved_snapshot_available:false,source_inventory_confirmed:false,declared_actor_scope_complete:false,windows:windows.map(w=>({window_id:w.window_id,status:'unstarted',declared_actor_count:0,labels:['raise_hand','stand_up','move_to_board','write_board'].map(label=>({label,reviewed_actor_seconds:0,declared_actor_seconds:0,coverage_fraction:null,unreviewed_actor_seconds:null,draft_intervals:0,uncertain_intervals:0,reviewed_positive_intervals:0,coverage_gaps:[],gaps_truncated:false}))}))},warnings:[]};
}
function install(s,value=fixture()){s.run(`pilotContext=${JSON.stringify(value)};renderPilotContext();`);}
test('empty branch displays six unstarted source windows and never treats absent humans as reviewed zero',()=>{
 const s=setup();install(s);
 assert.equal(s.get('pilotPanel').hidden,false);assert.equal(s.get('pilotWindows').children.length,6);
 assert.equal(s.get('pilotWindows').children.filter(n=>s.text(n).includes('Chưa bắt đầu')).length,6);
 assert.match(s.get('pilotState').textContent,/người đã khai báo|người do nhóm khai báo/);
 assert.match(s.get('pilotState').textContent,/chưa.*toàn bộ|chưa xác nhận/i);
 s.get('pilotWindows').children[0].listeners.click();
 assert.match(s.text(s.get('pilotAuditRows')),/Chưa có người/);assert.doesNotMatch(s.text(s.get('pilotAuditRows')),/100\s*%/);
});
test('explicit choice seeks exact saved time and clips defaults to window, actor and scene without creating labels',()=>{
 const s=setup(),p=fixture();s.run(`sceneSegments=[{segment_id:2,start_s:9,end_s:14}];segments=[{segment_id:2,track_id:'human:one',start_s:10,end_s:15}];$('actor').value=actorKey(segments[0]);$('humanScene').value='0';events=[{label:'unknown',start_s:2,end_s:3}];coverage=[];dirty=true;`);install(s,p);
 const before=s.run('JSON.stringify({events,coverage,dirty})');s.get('pilotWindows').children[1].listeners.click();
 assert.equal(s.get('video').currentTime,p.pilot.windows[1].start_s);
 assert.equal(s.get('startTime').value,'10');assert.equal(s.get('endTime').value,'14');
 assert.equal(s.get('humanStart').value,String(p.pilot.windows[1].start_s>9?p.pilot.windows[1].start_s:9));assert.equal(s.get('humanEnd').value,'14');
 assert.equal(s.run('JSON.stringify({events,coverage,dirty})'),before);
 assert.match(s.get('pilotSelection').textContent,/PTS/);
});
test('window navigation preserves an active edit and disjoint actor gets no out-of-window defaults',()=>{
 const s=setup();install(s);s.run(`edit={kind:'event',index:0};$('startTime').value='1';$('endTime').value='2';$('note').value='keep editing';$('review').value='reviewed';`);
 s.get('pilotWindows').children[3].listeners.click();
 assert.equal(s.get('startTime').value,'1');assert.equal(s.get('endTime').value,'2');assert.equal(s.get('note').value,'keep editing');assert.equal(s.get('review').value,'reviewed');assert.equal(s.run('edit.index'),0);
 assert.match(s.get('pilotSelection').textContent,/đang sửa|giữ.*sửa/i);
 s.run(`edit=null;segments=[{segment_id:1,track_id:'human:one',start_s:0,end_s:2}];sceneSegments=[{segment_id:1,start_s:0,end_s:48}];$('actor').value=actorKey(segments[0]);`);
 s.get('pilotWindows').children[4].listeners.click();assert.equal(s.get('startTime').value,'');assert.equal(s.get('endTime').value,'');
 assert.match(s.get('pilotSelection').textContent,/không giao|không có khoảng/i);
});
test('saved partial coverage and positive availability stay distinct from unreviewed scope and dirty edits',()=>{
 const s=setup(),p=fixture();p.review_snapshot='review-saved.json';p.audit.saved_snapshot_available=true;p.audit.windows[0].status='partial';p.audit.windows[0].declared_actor_count=1;
 Object.assign(p.audit.windows[0].labels[0],{reviewed_actor_seconds:3,declared_actor_seconds:8,coverage_fraction:.375,unreviewed_actor_seconds:5,reviewed_positive_intervals:0,draft_intervals:2,uncertain_intervals:1,coverage_gaps:[{segment_id:0,track_id:'human:one',start_s:3,end_s:8}]});
 s.run('evaluationReady=true;dirty=true;');install(s,p);s.get('pilotWindows').children[0].listeners.click();
 const rendered=s.text(s.get('pilotAuditRows'));assert.match(rendered,/37\.5/);assert.match(rendered,/Giơ tay/);assert.match(rendered,/Nháp.*2/);assert.match(rendered,/Chưa chắc.*1/);assert.match(rendered,/dương tính.*0/);
 assert.match(s.get('pilotState').textContent,/chưa lưu/);assert.equal(s.run('evaluationReady'),true);
 assert.equal(s.get('pilotWindows').children.filter(n=>s.text(n).includes('Chưa bắt đầu')).length,5);
});
test('refresh is branch scoped, aborts superseded requests and rejects late or wrong-identity responses',async()=>{
 const s=setup();s.run(`let pending=[],paths=[],signals=[];api=(path,options)=>{paths.push(path);signals.push(options.signal);return new Promise(resolve=>pending.push(resolve));};`);
 const old=s.run('refreshPilotContext()'),current=s.run('refreshPilotContext()');assert.equal(s.run('signals[0].aborted'),true);assert.match(s.run('paths[0]'),/pilot-context\?reviewer_id=group-a$/);
 s.run(`pending[1](${JSON.stringify(fixture())});`);await current;s.run(`pending[0]({...${JSON.stringify(fixture())},pilot:null,audit:null});`);await old;assert.equal(s.run('pilotContext.pilot.windows.length'),6);
 const bad=s.run('refreshPilotContext()');s.run(`pending[2]({...${JSON.stringify(fixture())},source_sha256:'wrong'});`);await bad;assert.equal(s.run('pilotContext'),null);assert.match(s.get('pilotState').textContent,/khớp|nguồn/);
 const other=s.run('refreshPilotContext()');s.run(`jobEpoch++;job={id:'two'};pending[3](${JSON.stringify(fixture())});`);await other;assert.equal(s.run('pilotContext'),null);
});
test('refresh validates branch and namespace, preserves drafts and hides nonpilot jobs',async()=>{
 const s=setup();s.run(`events=[{label:'LOCAL_UNSAVED'}];coverage=[];dirty=true;api=async()=>(${JSON.stringify(fixture())});`);await s.run('refreshPilotContext()');assert.equal(s.run('events[0].label'),'LOCAL_UNSAVED');assert.equal(s.run('dirty'),true);
 for(const override of [{review_branch:'group-b'},{actor_namespace:'run:other'},{job_id:'other'}]){s.run(`api=async()=>({...${JSON.stringify(fixture())},...${JSON.stringify(override)}});`);await s.run('refreshPilotContext()');assert.equal(s.run('pilotContext'),null);}
 s.run(`api=async()=>({...${JSON.stringify(fixture())},pilot:null,audit:null});`);await s.run('refreshPilotContext()');assert.equal(s.get('pilotPanel').hidden,true);
});
test('review load and successful save refresh pilot audit without blocking or replacing review state',async()=>{
 const s=setup(),context={review_branch:'group-a',source_sha256:hash,actor_namespace:namespace,duration_s:48,events:[],coverage:[],human_actors:[],track_segments:[],review_metadata:{reviewer_id:'group-a'}};
 s.run(`let pilotLoads=0;refreshPilotContext=()=>{pilotLoads++;return Promise.resolve();};job.source={duration_seconds:48};loadFrameTimes=async()=>false;renderArtifacts=()=>{};renderActors=()=>{};renderAnnotations=()=>{};updateControls=()=>{};updateSaveState=()=>{};readClassReviewContext=async()=>(${JSON.stringify(context)});`);
 await s.run('loadComplete(jobEpoch)');assert.equal(s.run('pilotLoads'),1);
 s.run(`post=async()=>(${JSON.stringify(context)});refreshClassSummary=async()=>{};`);await s.run('saveAnnotations()');assert.equal(s.run('pilotLoads'),2);
 s.run(`post=async()=>{throw Error('save rejected');};`);await s.run('saveAnnotations()');assert.equal(s.run('pilotLoads'),2);
});
test('clearing viewer aborts and clears pilot navigation and stale selection',async()=>{
 const s=setup();install(s);s.get('pilotWindows').children[0].listeners.click();s.run(`let signal;api=(_path,options)=>{signal=options.signal;return new Promise(()=>{});};refreshPilotContext();stopLiveReplay=()=>{};setReplayViewer=()=>{};drawOverlay=()=>{};renderClassSummary=()=>{};clearViewer();`);
 assert.equal(s.run('signal.aborted'),true);assert.equal(s.run('pilotContext'),null);assert.equal(s.run('selectedPilotWindow'),null);assert.equal(s.get('pilotPanel').hidden,true);
});
test('background audit refresh never seeks the video or replaces selected-window form edits',async()=>{
 const s=setup();install(s);s.get('pilotWindows').children[1].listeners.click();
 s.run(`video.currentTime=19.75;$('startTime').value='11.25';$('endTime').value='12.75';$('humanStart').value='11';$('humanEnd').value='13';$('note').value='unsaved note';api=async()=>(${JSON.stringify(fixture())});`);
 await s.run('refreshPilotContext()');assert.equal(s.get('video').currentTime,19.75);assert.equal(s.get('startTime').value,'11.25');assert.equal(s.get('endTime').value,'12.75');assert.equal(s.get('humanStart').value,'11');assert.equal(s.get('humanEnd').value,'13');assert.equal(s.get('note').value,'unsaved note');
 assert.equal(s.run('selectedPilotWindow'),'pilot-1');
});
test('complete declared scope remains separate from whole-source human inventory and GT export eligibility',()=>{
 const s=setup(),p=fixture();p.audit.declared_actor_scope_complete=true;for(const w of p.audit.windows)w.status='reviewed_declared_scope';s.run('evaluationReady=false;');install(s,p);
 assert.match(s.get('pilotState').textContent,/không đồng nghĩa hoàn tất pilot/);assert.match(s.get('pilotState').textContent,/chưa xác nhận toàn bộ/);assert.equal(s.run('evaluationReady'),false);
 assert.equal(s.get('pilotWindows').children.filter(n=>s.text(n).includes('Đã phủ người đã khai báo')).length,6);
 const before=s.get('video').currentTime;s.run('reviewReady=false;');s.get('pilotWindows').children[0].listeners.click();assert.equal(s.get('video').currentTime,before);assert.equal(s.run('selectedPilotWindow'),null);
});
