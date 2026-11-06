const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict');
const {test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
function setup(){
 const nodes=new Map();
 const node=()=>({value:'',dataset:{},style:{},children:[],get options(){return this.children;},listeners:{},checked:false,hidden:false,textContent:'',append(...items){this.children.push(...items);},replaceChildren(...items){this.children=[...items];this.textContent='';},addEventListener(name,fn){this.listeners[name]=fn;},setAttribute(){},getContext(){return {clearRect(){}};},querySelector(){return node();},pause(){},load(){},removeAttribute(){}});
 const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
 get('classLabel').value='raise_hand';get('classWindowSeconds').value='60';
 const context=vm.createContext({document:{getElementById:get,createElement:node},Headers,URLSearchParams,AbortController,TextDecoder,Uint8Array,Float64Array,DataView,localStorage:{getItem(){return null;}}});
 vm.runInContext(html.split('<script>')[1].split('if(videoFrameAPI){')[0],context);
 const run=code=>vm.runInContext(code,context);
 run(`job={id:'one',state:'complete',preview_verified:true};config={};`);
 const text=n=>[n.textContent,...n.children.map(text)].join(' ');
 return {get,run,text};
}
const metric=(count,duration,coverage)=>({count,duration_s:duration,reviewed_coverage_actor_s:coverage,truncated_episode_count:0});
function fixture(){return {schema_version:2,source_sha256:'a'.repeat(64),review_snapshot:'review-'+ 'b'.repeat(32)+'.json',duration_s:120,track_segments:[{segment_id:0,track_id:1,start_s:0,end_s:120}],events:[{segment_id:0,track_id:1,label:'raise_hand',start_s:65,end_s:70,review:'reviewed',origin:'human',uncertain:false}],coverage:[],class_summary:{session:{raise_hand:metric(0,0,60)},saved_intervals:{unknown:1,uncertain:1,draft:1},windows:[{start_s:0,end_s:60,labels:{raise_hand:metric(null,null,0)},saved_intervals:{unknown:1,uncertain:1,draft:1}},{start_s:60,end_s:120,labels:{raise_hand:metric(0,0,60)},saved_intervals:{unknown:0,uncertain:0,draft:0}}]}};}
test('class view distinguishes no reviewed coverage from zero and keeps performance report separate',()=>{
 const s=setup();s.get('summary').textContent='pipeline performance';
 s.run(`classSummary=${JSON.stringify(fixture())};renderClassSummary();`);
 assert.match(s.text(s.get('classWindowRows').children[0]),/Chưa rà soát/);
 assert.match(s.text(s.get('classWindowRows').children[1]),/0/);
 assert.match(s.text(s.get('classSummaryState')),/đã lưu/);
 assert.match(s.text(s.get('classSession')),/Không xác định.*1/);
 assert.match(s.text(s.get('classSession')),/Chưa chắc.*1/);
 assert.equal(s.get('summary').textContent,'pipeline performance');
 s.run('dirty=true;updateClassSummaryState();');assert.match(s.text(s.get('classSummaryState')),/chưa lưu/);
});
test('class evidence seeks original event time and selects the containing local actor extent',()=>{
 const s=setup();s.run(`classSummary=${JSON.stringify(fixture())};segments=classSummary.track_segments;let changed=0;actorChanged=()=>changed++;requestFrame=()=>{};showClassEvidence(1);`);
 const button=s.get('classEvidence').children.find(n=>typeof n.listeners.click==='function');
 assert.ok(button,'saved event must be available as a source-time evidence link');button.listeners.click();
 assert.equal(s.get('video').currentTime,65);assert.equal(s.run('changed'),1);
 assert.equal(s.get('actor').value,s.run('actorKey(segments[0])'));
});
test('completion reload restores saved class summary without replacing pipeline performance data',async()=>{
 const s=setup();const payload=fixture();
 s.run(`job.source={duration_seconds:120};job.summary={elapsed_seconds:12};let paths=[];api=async path=>{paths.push(path);return ${JSON.stringify(payload)};};loadFrameTimes=async()=>false;renderArtifacts=()=>{};renderActors=()=>{};renderAnnotations=()=>{};updateControls=()=>{};updateSaveState=()=>{};`);
 await s.run('loadComplete(jobEpoch)');
 assert.ok(s.run('paths.some(p=>p.includes("/class-summary"))'));
 assert.equal(s.run('classSummary.review_snapshot'),payload.review_snapshot);
 assert.equal(s.get('summary').textContent,JSON.stringify({elapsed_seconds:12},null,2));
 assert.match(s.text(s.get('classSession')),/Giơ tay/);
});
test('an invalid summary window does not block restoring the saved review context',async()=>{
 const s=setup();const payload=fixture();delete payload.class_summary;s.get('classWindowSeconds').value='0.000001';
 s.run(`job.source={duration_seconds:120};api=async path=>{if(path.includes('/class-summary'))throw Error('Window too small');return ${JSON.stringify(payload)};};loadFrameTimes=async()=>false;renderArtifacts=()=>{};renderActors=()=>{};renderAnnotations=()=>{};updateControls=()=>{};updateSaveState=()=>{};`);
 await s.run('loadComplete(jobEpoch)');
 assert.equal(s.run('reviewReady'),true);assert.equal(s.run('classSummary'),null);
 assert.match(s.text(s.get('classSummaryState')),/Window too small/);
});
test('saving annotations refreshes persisted class data and keeps the pipeline report intact',async()=>{
 const s=setup();s.get('summary').textContent='pipeline performance';
 s.run(`reviewReady=true;events=[];coverage=[];let requests=[];post=async()=>({events:[],coverage:[],summary:{session:{}},snapshot:'review-${'b'.repeat(32)}.json'});api=async path=>{requests.push(path);return ${JSON.stringify(fixture())};};renderArtifacts=()=>{};renderAnnotations=()=>{};updateControls=()=>{};`);
 await s.run('saveAnnotations()');
 assert.equal(s.run('classSummary.review_snapshot'),fixture().review_snapshot);
 assert.ok(s.run('requests.some(p=>p.includes("/class-summary"))'));
 assert.equal(s.get('summary').textContent,'pipeline performance');
 assert.equal(s.run('dirty'),false);
});
test('late class-summary responses cannot replace a newer window request or another job',async()=>{
 const s=setup();s.run(`let pending=[];api=()=>new Promise(resolve=>pending.push(resolve));`);
 const old=s.run('refreshClassSummary()'),current=s.run('refreshClassSummary()');
 s.run(`pending[1](${JSON.stringify(fixture())});`);await current;
 s.run(`pending[0]({review_snapshot:'obsolete',class_summary:{}});`);await old;
 assert.equal(s.run('classSummary.review_snapshot'),fixture().review_snapshot);
 const other=s.run('refreshClassSummary()');s.run('jobEpoch++;classSummary=null;pending[2]({review_snapshot:"old-job"});');await other;
 assert.equal(s.run('classSummary'),null);
});
test('saving an older job cannot attach its snapshot after switching jobs during summary refresh',async()=>{
 const s=setup();
 s.run(`reviewReady=true;events=[];coverage=[];let resolveSummary;post=async()=>({events:[],coverage:[],snapshot:'review-${'b'.repeat(32)}.json'});api=()=>new Promise(resolve=>{resolveSummary=resolve;});renderArtifacts=()=>{};renderAnnotations=()=>{};updateControls=()=>{};`);
 const saving=s.run('saveAnnotations()');await Promise.resolve();
 s.run(`jobEpoch++;job={id:'two',state:'complete',artifacts:[]};resolveSummary(${JSON.stringify(fixture())});`);await saving;
 assert.equal(s.run('job.artifacts.length'),0);
});

test('class export downloads only the saved aggregate JSON and rejects unknown schema versions',async()=>{
 const s=setup();s.get('classWindowSeconds').value='15';
 s.run(`let exportCalls=[],exported=[],downloads=0;Blob=class{constructor(parts){exported.push(parts.join(''));}};URL={createObjectURL(){return 'blob:local';},revokeObjectURL(){}};setTimeout=()=>{};document.createElement=()=>({click(){downloads++;}});api=async path=>{exportCalls.push(path);return {schema_version:1,kind:'class_observation_summary',class_session:{raise_hand:{count:null}}};};`);
 await s.run('downloadClassExport()');
 assert.equal(s.run('downloads'),1);assert.equal(s.run('exportCalls.length'),1);
 assert.match(s.run('exportCalls[0]'),/\/class-export\?window_seconds=15$/);
 assert.equal(JSON.parse(s.run('exported[0]')).class_session.raise_hand.count,null);
 s.run(`api=async()=>({schema_version:99,kind:'class_observation_summary'});`);
 await s.run('downloadClassExport()');
 assert.equal(s.run('downloads'),1);assert.match(s.get('errorText').textContent,/phiên bản|schema/);
});

test('loading a new reviewer branch ignores legacy and other-reviewer drafts and restores only its own draft',async()=>{
 const s=setup(),context=fixture();context.review_branch='group-b';context.review_snapshot=null;context.events=[];context.coverage=[];context.human_actors=[];context.review_metadata={reviewer_id:'group-b',adjudication_state:'draft'};context.actor_namespace='run:pilot';
 const otherDraft={dirty:true,events:[{label:'PRIVATE_A_LABEL'}],coverage:[],review_metadata:{reviewer_id:'group-a'}};
 s.run(`activeReviewBranch='group-b';job.source={duration_seconds:120};let stored={'classroom-demo-draft-one':${JSON.stringify(JSON.stringify(otherDraft))},'classroom-demo-draft-one-reviewer-group-a':${JSON.stringify(JSON.stringify(otherDraft))}},draftKeys=[];storageGet=key=>{draftKeys.push(key);return stored[key]||null;};api=async()=>(${JSON.stringify(context)});loadFrameTimes=async()=>false;renderArtifacts=()=>{};renderActors=()=>{};renderAnnotations=()=>{};updateControls=()=>{};updateSaveState=()=>{};`);
 await s.run('loadComplete(jobEpoch)');
 assert.equal(s.run('events.length'),0);assert.equal(s.run('humanActors.length'),0);
 assert.equal(s.run('draftKeys[0]'),'classroom-demo-draft-one-reviewer-group-b');
 assert.equal(s.run('readReviewMetadata().reviewer_id'),'group-b');
 const ownDraft={...otherDraft,events:[{label:'stand_up'}],review_metadata:{reviewer_id:'group-b'}};
 s.run(`stored['classroom-demo-draft-one-reviewer-group-b']=${JSON.stringify(JSON.stringify(ownDraft))};`);
 await s.run('loadComplete(jobEpoch)');
 assert.equal(s.run('events[0].label'),'stand_up');assert.equal(s.run('dirty'),true);
});

test('review readiness explains missing human inputs in Vietnamese without exposing raw API codes or changing them',()=>{
 const s=setup();s.run(`readinessReasons=['missing_rubric_version','missing_independent_human_actors','future_internal_code'];evaluationReady=false;dirty=false;renderActorReview();`);
 assert.match(s.get('reviewReadiness').textContent,/phiên bản rubric/);
 assert.match(s.get('reviewReadiness').textContent,/người quan sát độc lập/);
 assert.doesNotMatch(s.get('reviewReadiness').textContent,/missing_|future_internal_code/);
 assert.equal(s.run('readinessReasons[0]'),'missing_rubric_version');
 assert.equal(s.run('evaluationReady'),false);
});
