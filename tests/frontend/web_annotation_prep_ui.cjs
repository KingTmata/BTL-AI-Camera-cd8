const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict'),{test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
const id='a'.repeat(32),hash='b'.repeat(64),human='human:'+ 'c'.repeat(32);
const frame=index=>({frame_index:index,source_pts:index*100,time_base:'1/1000',timestamp_seconds:index/10,segment_id:'source',tracks:[],display_tracks:[],detections:[]});
const actor={segment_id:'source',track_id:human,start_s:0,end_s:1};
const review=()=>({source_sha256:hash,actor_namespace:'source:'+hash,review_snapshot:null,review_branch:null,duration_s:1,events:[],coverage:[],human_actors:[actor],human_actor_references:[],actor_links:[],model_track_segments:[],scene_segments:[{segment_id:'source',start_s:0,end_s:1}],review_metadata:{},evaluation_ready:false,readiness_reasons:['missing_actor_reference']});
function setup(){
 const nodes=new Map(),calls=[],navigation=[],stored=[];
 const ctx=new Proxy({}, {get:(_,name)=>(...args)=>{calls.push([name,...args]);return name==='measureText'?{width:100}:undefined;},set:()=>true});
 const node=()=>({value:'',checked:false,hidden:false,disabled:false,dataset:{},style:{},parentElement:{hidden:false},children:[],textContent:'',listeners:{},get options(){return this.children;},append(...items){this.children.push(...items);if(!this.value&&items[0]?.value)this.value=items[0].value;},replaceChildren(...items){this.children=items;this.value='';this.textContent='';},addEventListener(name,fn){this.listeners[name]=fn;},setAttribute(name,value){this[name]=value;},removeAttribute(name){delete this[name];},querySelector(){return node();},querySelectorAll(){return [];},getContext(){return ctx;},getBoundingClientRect(){return {left:100,top:200,width:500,height:250};},pause(){this.paused=true;},load(){},focus(){}});
 const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
 const context=vm.createContext({document:{getElementById:get,createElement:node},window:{confirm:()=>false},location:{href:`http://127.0.0.1:8090/?job=${id}&reviewer=group-a&diagnostics=1`,assign:url=>navigation.push(url)},Headers,URL,URLSearchParams,AbortController,TextDecoder,Uint8Array,Float64Array,DataView,setTimeout:()=>1,clearTimeout(){}});
 vm.runInContext(html.split('<script>')[1].split('if(videoFrameAPI){')[0],context);
 vm.runInContext(html.split(/\r?\n/).find(line=>line.startsWith('function openReviewerBranch(')),context);
 const run=code=>vm.runInContext(code,context),text=n=>[n.textContent,...n.children.map(text)].join(' ');
 run(`job={id:'${id}',state:'complete',processing_mode:'annotation_prep',kind:'annotation_source_preparation',preview_verified:true,inference_status:'not_run',source_sha256:'${hash}',source:{width:1000,height:500,frame_count:10,duration_seconds:1},video_export_eligible:false,artifacts:[]};config={token:'local',models:[],pose_edges13:[]};source=job.source;reviewReady=true;frameTimes=Float64Array.from({length:10},(_,i)=>i/10);currentFrame=${JSON.stringify(frame(5))};currentFrameIndex=5;requestedFrame=5;video.currentTime=.5;video.paused=true;video.videoWidth=1000;video.videoHeight=500;$('overlay').width=1000;$('overlay').height=500;storageSet=(key,value)=>{};restoreReview(${JSON.stringify(review())});$('actor').value=actorKey(humanActors[0]);requestFrame=()=>{};`);
 run('storageSet=(key,value)=>stored.push({key,value});');context.stored=stored;
 return {get,run,text,calls,navigation,stored};
}
const anchor=()=>({segment_id:actor.segment_id,track_id:actor.track_id,frame_index:5,source_pts:500,time_base:'1/1000',timestamp_seconds:.5,bbox_xyxyn:[.1,.2,.5,.8],note:'áo sáng, bên trái'});
test('annotation preparation disables AI settings and submits only source to its dedicated endpoint',async()=>{
 const s=setup();s.run(`job=null;$('processingMode').value='annotation_prep';$('model').value='';$('jobForm').reportValidity=()=>true;$('source').value='clip.mov';$('file').files=[];updateControls();`);
 for(const key of ['model','device','task','tracking','confidence','imgsz','nms','sampleFPS','actionRules','saveActionVideo'])assert.equal(s.get(key).disabled,true,key);
 assert.equal(s.get('start').disabled,false);s.run(`let requests=[];post=async(path,body)=>{requests.push({path,body});return {id:'${'d'.repeat(32)}',processing_mode:'annotation_prep'};};clearViewer=()=>{};renderAnnotations=()=>{};renderArtifacts=()=>{};renderJob=()=>{};poll=()=>{};updateControls=()=>{};`);
 await s.run('startJob({preventDefault(){}})');assert.equal(s.run('requests[0].path'),'/api/annotation-sources');assert.deepEqual(JSON.parse(s.run('JSON.stringify(requests[0].body)')),{source:'clip.mov'});
});
test('prepared source review stays human-enabled, suppresses model overlays/proposals/links/export and never prepares action rules',async()=>{
 const s=setup();s.run(`$('showActions').checked=true;let calls=[];api=async path=>{calls.push(path);throw Error('AI forbidden');};updateControls();updateAnnotationViewer();`);await s.run('prepareActionCaptions()');
 assert.equal(s.run('calls.length'),0);assert.equal(s.get('reviewFields').disabled,false);assert.equal(s.get('proposals').disabled,true);assert.equal(s.get('modelActorLinks').hidden,true);assert.equal(s.get('proposalNote').hidden,true);assert.equal(s.get('videoExportPanel').hidden,true);assert.equal(s.get('showActions').disabled,true);
 s.run(`frameCache.set(5,currentFrame);showCachedFrame();`);assert.match(s.get('frameStatus').textContent,/nguồn chưa chạy AI/i);
});
test('prepared-source selector validates review URL and preserves a dirty draft until the user chooses to switch',async()=>{
 const s=setup(),next='d'.repeat(32);s.run(`api=async()=>({sources:[{job_id:'${next}',name:'Nguồn đã chuẩn bị',source_sha256:'${hash}',state:'complete',duration_seconds:1,review_url:'/?job=${next}'}]});`);await s.run('refreshPreparedSources()');s.get('preparedSource').value=next;s.run(`dirty=true;window.confirm=()=>false;openPreparedSource();`);assert.equal(s.navigation.length,0);
 s.run(`window.confirm=()=>true;openPreparedSource();`);assert.equal(s.navigation.length,1);const url=new URL(s.navigation[0]);assert.equal(url.searchParams.get('job'),next);assert.equal(url.searchParams.get('reviewer'),'group-a');assert.equal(url.searchParams.get('diagnostics'),'1');assert.equal(s.stored.length,1);
 s.run(`api=async()=>({sources:[{job_id:'${next}',name:'bad',review_url:'https://outside.invalid/?job=${next}'}]});`);await s.run('refreshPreparedSources()');assert.equal(s.run('preparedSources.size'),0);
});
test('superseded prepared catalog responses are aborted and cannot replace the newer selector contents',async()=>{
 const s=setup();s.run(`let pending=[];api=(_path,options)=>new Promise(resolve=>pending.push({resolve,signal:options.signal}));`);const old=s.run('refreshPreparedSources()'),latest=s.run('refreshPreparedSources()');assert.equal(s.run('pending[0].signal.aborted'),true);
 s.run(`pending[1].resolve({sources:[]});`);await latest;s.run(`pending[0].resolve({sources:[{job_id:'${'d'.repeat(32)}',name:'OLD',review_url:'/?job=${'d'.repeat(32)}'}]});`);await old;assert.equal(s.run('preparedSources.size'),0);
});
test('draft rubric guide renders canonical definitions and pre-fills only a new empty preparation branch',async()=>{
 const s=setup(),rubric={version:'classroom-actions-v1-draft-20261103',status:'draft',labels:[{id:'raise_hand',title:'Giơ tay',kind:'action',definition:'Chuỗi có cổ tay cao hơn vai',onset:'Bắt đầu quan sát',end:'Hạ tay',uncertain:'Thiếu khớp',notes:'Có thể chồng hành động khác'}],principles:['Bốn hành động độc lập; coverage chưa rà khác âm.'],source_reference:{archive_sha256:'e'.repeat(64)}};
 s.run(`api=async()=>(${JSON.stringify(rubric)});`);await s.run('loadActionRubric()');assert.match(s.text(s.get('rubricLabels')),/Chuỗi có cổ tay/);assert.match(s.get('rubricState').textContent,/draft|nháp/i);assert.equal(s.get('rubricVersion').value,rubric.version);assert.equal(s.run('dirty'),false);
 s.run(`restoreReview({...${JSON.stringify(review())},review_snapshot:'saved.json',review_metadata:{rubric_version:'teacher-existing'}});`);await s.run('loadActionRubric()');assert.equal(s.get('rubricVersion').value,'teacher-existing');
 s.run(`restoreReview({...${JSON.stringify(review())},dirty:true,review_metadata:{rubric_version:''}});`);assert.equal(s.get('rubricVersion').value,'');
});
test('human reference captures exact paused source frame and normalized box into local draft only',()=>{
 const s=setup();s.get('humanReferenceNote').value='áo sáng, bên trái';s.run(`let posts=0;post=()=>posts++;beginHumanReference();referencePointerDown({clientX:150,clientY:250,preventDefault(){}});referencePointerUp({clientX:150,clientY:250,preventDefault(){}});referencePointerDown({clientX:350,clientY:400,preventDefault(){}});referencePointerUp({clientX:350,clientY:400,preventDefault(){}});saveHumanReferenceDraft();`);
 const value=JSON.parse(s.run('JSON.stringify(humanActorReferences[0])'));assert.deepEqual(value,anchor());assert.equal(s.run('posts'),0);assert.equal(s.run('dirty'),true);assert.equal(s.run('events.length'),0);assert.equal(s.run('coverage.length'),0);assert.deepEqual(JSON.parse(s.run('JSON.stringify(reviewPayload().human_actor_references[0])')),anchor());
});
test('reference requires an independently declared human and exact currently verified frame; invalid notes and duplicates do not mutate draft',()=>{
 const s=setup();s.run(`segments=[{segment_id:'source',track_id:7,start_s:0,end_s:1}];$('actor').value=actorKey(segments[0]);beginHumanReference();`);assert.equal(s.run('humanReferenceSelection'),null);assert.match(s.get('errorText').textContent,/người.*độc lập|do nhóm/i);
 s.run(`segments=humanActors;$('actor').value=actorKey(segments[0]);currentFrameIndex=6;beginHumanReference();`);assert.equal(s.run('humanReferenceSelection'),null);
 s.run(`currentFrameIndex=5;humanActorReferences=[${JSON.stringify(anchor())}];beginHumanReference();`);assert.equal(s.run('humanReferenceSelection'),null);assert.match(s.get('errorText').textContent,/đã có/);
});
test('saved anchors paint only their exact source frame and a seek invalidates a pending reference selection',()=>{
 const s=setup();s.run(`humanActorReferences=[${JSON.stringify(anchor())}];drawHumanReferences($('overlay').getContext('2d'),1000,500,1);`);assert.ok(s.calls.some(c=>c[0]==='strokeRect'&&c[1]===100&&c[2]===100&&c[3]===400&&Math.abs(c[4]-300)<1e-9));
 s.calls.length=0;s.run(`currentFrame=${JSON.stringify(frame(6))};currentFrameIndex=6;drawHumanReferences($('overlay').getContext('2d'),1000,500,1);`);assert.equal(s.calls.filter(c=>c[0]==='strokeRect').length,0);
 s.run(`humanActorReferences=[];currentFrame=${JSON.stringify(frame(5))};currentFrameIndex=5;beginHumanReference();currentFrame=${JSON.stringify(frame(6))};currentFrameIndex=6;drawHumanReferences($('overlay').getContext('2d'),1000,500,1);saveHumanReferenceDraft();`);assert.equal(s.run('humanActorReferences.length'),0);
 s.run(`currentFrame=${JSON.stringify(frame(5))};currentFrameIndex=5;humanActorReferences=[${JSON.stringify(anchor())}];jobEpoch++;`);s.calls.length=0;s.run(`drawHumanReferences($('overlay').getContext('2d'),1000,500,1);`);assert.equal(s.calls.filter(c=>c[0]==='strokeRect').length,0);
});
test('reference seek/delete and actor removal keep anchors within remaining declared spans without automatic tracking',()=>{
 const s=setup();s.run(`humanActorReferences=[${JSON.stringify(anchor())}];renderHumanReferences();seekHumanReference(0);`);assert.equal(s.get('video').currentTime,.5);assert.equal(s.get('actor').value,s.run('actorKey(humanActors[0])'));
 s.run('removeHumanReference(0)');assert.equal(s.run('humanActorReferences.length'),0);assert.equal(s.run('humanActors.length'),1);
 s.run(`humanActors.push({...humanActors[0],start_s:1,end_s:2});segments=humanActors;humanActorReferences=[${JSON.stringify(anchor())},{...${JSON.stringify(anchor())},frame_index:15,source_pts:1500,timestamp_seconds:1.5}];removeHumanActor(0);`);assert.equal(s.run('humanActorReferences.length'),1);assert.equal(s.run('humanActorReferences[0].frame_index'),15);
});
test('restoration retains reference schema and translates missing-reference readiness without inventing GT',()=>{
 const s=setup();s.run(`restoreReview({...${JSON.stringify(review())},human_actor_references:[${JSON.stringify(anchor())}]});`);assert.deepEqual(JSON.parse(s.run('JSON.stringify(humanActorReferences[0])')),anchor());assert.match(s.get('reviewReadiness').textContent,/tham chiếu/);assert.doesNotMatch(s.get('reviewReadiness').textContent,/missing_actor_reference/);assert.equal(s.run('evaluationReady'),false);
});


test('actual preparation completion restores review and native preview without fetching or preparing AI metadata',async()=>{
 const s=setup();s.run(`job.artifacts=['preview.mp4'];$('processingMode').value='full';$('showActions').checked=true;let aiCalls=0;prepareActionCaptions=()=>{aiCalls++;};loadBoardROI=()=>{aiCalls++;};loadFrameTimes=async()=>{frameTimes=Float64Array.from({length:10},(_,i)=>i/10);};primeFrames=async()=>{};readClassReviewContext=async()=>(${JSON.stringify(review())});refreshPilotContext=async()=>{};storageGet=()=>null;`);
 await s.run('loadComplete(jobEpoch)');assert.equal(s.run('aiCalls'),0);assert.equal(s.get('processingMode').value,'annotation_prep');assert.equal(s.get('video').src,`/api/jobs/${id}/artifacts/preview.mp4`);assert.equal(s.run('reviewReady'),true);assert.match(s.get('sourceFacts').textContent,/chưa chạy AI/);assert.equal(s.get('model').disabled,true);assert.equal(s.run('events.length'),0);
});

test('a seeking source image cannot begin or commit a reference and a foreign source hash cannot paint one',()=>{
 const s=setup();s.run(`video.seeking=true;beginHumanReference();`);assert.equal(s.run('humanReferenceSelection'),null);
 s.run(`video.seeking=false;beginHumanReference();referenceBox=[.1,.2,.5,.8];video.seeking=true;saveHumanReferenceDraft();`);assert.equal(s.run('humanActorReferences.length'),0);
 s.run(`video.seeking=false;job.source_sha256='${'f'.repeat(64)}';humanActorReferences=[${JSON.stringify(anchor())}];drawHumanReferences($('overlay').getContext('2d'),1000,500,1);`);assert.equal(s.calls.filter(c=>c[0]==='strokeRect').length,0);
});

test('reference note and count bounds reject draft mutation while edited annotation remains intact when seeking an anchor',()=>{
 const s=setup();s.run(`beginHumanReference();referenceBox=[.1,.2,.5,.8];$('humanReferenceNote').value='x'.repeat(241);saveHumanReferenceDraft();`);assert.equal(s.run('humanActorReferences.length'),0);assert.equal(s.run('dirty'),false);
 s.run(`$('humanReferenceNote').value='';humanActorReferences=Array.from({length:4000},()=>(${JSON.stringify(anchor())}));saveHumanReferenceDraft();`);assert.equal(s.run('humanActorReferences.length'),4000);assert.equal(s.run('dirty'),false);
 s.run(`humanActorReferences=[${JSON.stringify(anchor())}];edit={kind:'event',index:2};$('note').value='keep this edit';$('startTime').value='.25';seekHumanReference(0);`);assert.equal(s.get('note').value,'keep this edit');assert.equal(s.get('startTime').value,'.25');assert.equal(s.run('edit.index'),2);
});

test('preparation submit preserves an unsaved branch when source-switch confirmation is declined',async()=>{
 const s=setup();s.run(`dirty=true;$('processingMode').value='annotation_prep';$('jobForm').reportValidity=()=>true;let posts=0;post=async()=>{posts++;};window.confirm=()=>false;`);await s.run('startJob({preventDefault(){}})');assert.equal(s.run('posts'),0);assert.equal(s.run('dirty'),true);assert.equal(s.run('job.id'),id);
});


function prepareSwitch(s){const next='d'.repeat(32);s.run(`preparedSources.set('${next}',{job_id:'${next}',review_url:'/?job=${next}'});$('preparedSource').value='${next}';`);}
function savedEvent(s){s.run(`events=[{segment_id:'source',track_id:'${human}',label:'raise_hand',start_s:0,end_s:1,review:'reviewed',origin:'human',uncertain:false,evidence:{note:'saved observation'}}];dirty=false;editAnnotation('event',0);`);}

test('declining a source switch preserves an actually edited saved-event form even when committed arrays are clean',()=>{
 const s=setup();prepareSwitch(s);savedEvent(s);s.run(`$('note').value='not added yet';let confirmations=0;window.confirm=()=>{confirmations++;return false;};openPreparedSource();`);
 assert.equal(s.run('confirmations'),1);assert.equal(s.navigation.length,0);assert.equal(s.get('note').value,'not added yet');assert.equal(s.run('edit.index'),0);assert.equal(s.run('events[0].evidence.note'),'saved observation');assert.equal(s.run('dirty'),false);assert.equal(s.stored.length,0);
});

test('an unchanged opened event needs no warning, but a new partial annotation form does',()=>{
 const untouched=setup();prepareSwitch(untouched);savedEvent(untouched);untouched.run(`let confirmations=0;window.confirm=()=>{confirmations++;return false;};openPreparedSource();`);assert.equal(untouched.run('confirmations'),0);assert.equal(untouched.navigation.length,1);
 const s=setup();prepareSwitch(s);s.run(`actorChanged();$('startTime').value='.2';$('note').value='new partial event';let confirmations=0;window.confirm=()=>{confirmations++;return false;};openPreparedSource();`);assert.equal(s.run('confirmations'),1);assert.equal(s.navigation.length,0);assert.equal(s.get('startTime').value,'.2');assert.equal(s.run('events.length'),0);
});

test('declining source switch preserves an uncommitted reference selection and box without adding it to annotations',()=>{
 const s=setup();prepareSwitch(s);s.run(`beginHumanReference();referenceBox=[.1,.2,.5,.8];referenceDrawing=false;$('humanReferenceNote').value='pending box';let confirmations=0;window.confirm=()=>{confirmations++;return false;};openPreparedSource();`);assert.equal(s.run('confirmations'),1);assert.equal(s.navigation.length,0);assert.equal(s.run('humanReferenceSelection.frame_index'),5);assert.deepEqual(JSON.parse(s.run('JSON.stringify(referenceBox)')),[.1,.2,.5,.8]);assert.equal(s.get('humanReferenceNote').value,'pending box');assert.equal(s.run('humanActorReferences.length'),0);assert.equal(s.stored.length,0);
});

test('accepting a source switch persists only committed draft arrays and warns that uncommitted form or box will be discarded',()=>{
 const s=setup();prepareSwitch(s);savedEvent(s);s.run(`dirty=true;$('note').value='unfinished replacement';beginHumanReference();referenceBox=[.1,.2,.5,.8];let prompt='';window.confirm=message=>{prompt=message;return true;};openPreparedSource();`);assert.equal(s.navigation.length,1);assert.match(s.run('prompt'),/chưa.*(?:thêm|lưu)|biểu mẫu/i);assert.match(s.run('prompt'),/bỏ|mất/i);assert.equal(s.stored.length,1);const draft=JSON.parse(s.stored[0].value);assert.equal(draft.events[0].evidence.note,'saved observation');assert.deepEqual(draft.human_actor_references,[]);assert.equal(draft.dirty,true);
});


test('server save rejects uncommitted edited fields or reference boxes without posting or clearing them',async()=>{
 const s=setup();savedEvent(s);s.run(`$('note').value='edited but not added';let posts=0;post=async()=>{posts++;return ${JSON.stringify(review())};};refreshClassSummary=async()=>{};refreshPilotContext=async()=>{};`);await s.run('saveAnnotations()');assert.equal(s.run('posts'),0);assert.equal(s.get('note').value,'edited but not added');assert.equal(s.run('edit.index'),0);assert.match(s.get('errorText').textContent,/thêm|lưu sửa|hủy/i);
 s.run(`resetEdit();beginHumanReference();referenceBox=[.1,.2,.5,.8];referenceDrawing=false;$('humanReferenceNote').value='keep box';`);await s.run('saveAnnotations()');assert.equal(s.run('posts'),0);assert.equal(s.run('humanReferenceSelection.frame_index'),5);assert.equal(s.get('humanReferenceNote').value,'keep box');assert.equal(s.run('humanActorReferences.length'),0);
});

test('an edited form or pending box started during a save request survives the successful server response',async()=>{
 for(const reference of [false,true]){const s=setup();savedEvent(s);s.run(`resetEdit();let finishSave;post=()=>new Promise(resolve=>finishSave=resolve);refreshClassSummary=async()=>{};refreshPilotContext=async()=>{};`);const saving=s.run('saveAnnotations()');
 if(reference)s.run(`beginHumanReference();referenceBox=[.1,.2,.5,.8];referenceDrawing=false;$('humanReferenceNote').value='new pending box';`);else s.run(`editAnnotation('event',0);$('note').value='typed during request';`);
 s.run(`finishSave(${JSON.stringify(review())});`);await saving;if(reference){assert.equal(s.run('humanReferenceSelection?.frame_index'),5);assert.equal(s.get('humanReferenceNote').value,'new pending box');}else{assert.equal(s.get('note').value,'typed during request');assert.equal(s.run('edit?.index'),0);}
 assert.equal(s.run('humanActorReferences.length'),0);
 }
});


test('reviewer-branch navigation uses the same pending-form guard and decline keeps the open event intact',()=>{
 const s=setup();savedEvent(s);s.run(`$('reviewerId').value='group-b';$('note').value='still editing';let confirmations=0;window.confirm=()=>{confirmations++;return false;};openReviewerBranch();`);assert.equal(s.run('confirmations'),1);assert.equal(s.navigation.length,0);assert.equal(s.get('note').value,'still editing');assert.equal(s.run('edit.index'),0);
 s.run(`window.confirm=()=>true;openReviewerBranch();`);assert.equal(new URL(s.navigation[0]).searchParams.get('reviewer'),'group-b');assert.equal(s.stored.length,0);
});

test('a partial new annotation can be explicitly cancelled after a blocked server save',async()=>{
 const s=setup();s.run(`actorChanged();$('note').value='partial new event';let posts=0;post=async()=>{posts++;return ${JSON.stringify(review())};};refreshClassSummary=async()=>{};refreshPilotContext=async()=>{};`);await s.run('saveAnnotations()');assert.equal(s.run('posts'),0);assert.equal(s.get('resetEdit').hidden,false);assert.equal(s.get('note').value,'partial new event');s.run('resetEdit()');assert.equal(s.run('pendingReviewInput()'),false);assert.equal(s.run('events.length'),0);
});
