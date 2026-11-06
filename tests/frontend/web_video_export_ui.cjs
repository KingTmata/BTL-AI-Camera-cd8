const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict'),{test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
const jobID='a'.repeat(32),exportID='b'.repeat(32),hash='c'.repeat(64);
function operation(state='running') {return {id:exportID,state,error:null,progress:{phase:'rendering',frames_written:4,frames_total:10,source_time_s:2},options:{start_s:0,end_s:20,action_rules:'basic_pose',board_roi:null,board_roi_source_sha256:null},video_url:state==='complete'?`/api/jobs/${jobID}/video-export/${exportID}/video.mp4?inline=true`:null,report_url:state==='complete'?`/api/jobs/${jobID}/video-export/${exportID}/report.json`:null};}
function setup(){
 const nodes=new Map(),timers=[];let timerID=0;
 const node=()=>({value:'',checked:false,disabled:false,hidden:false,dataset:{},style:{},children:[],textContent:'',listeners:{},get options(){return this.children;},append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},addEventListener(name,fn){this.listeners[name]=fn;},setAttribute(name,value){this[name]=value;},removeAttribute(name){delete this[name];},querySelector(){return node();},querySelectorAll(){return [];},getContext(){return {clearRect(){}};},pause(){},load(){}});
 const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
 const context=vm.createContext({document:{getElementById:get,createElement:node},Headers,URL,URLSearchParams,AbortController,TextDecoder,Uint8Array,Float64Array,DataView,setTimeout(fn,delay){const id=++timerID;timers.push({id,fn,delay});return id;},clearTimeout(id){const i=timers.findIndex(t=>t.id===id);if(i>=0)timers.splice(i,1);}});
 vm.runInContext(html.split('<script>')[1].split('if(videoFrameAPI){')[0],context);
 vm.runInContext(html.split(/\r?\n/).find(line=>line.startsWith('function initialJobID(')),context);
 const run=code=>vm.runInContext(code,context);
 run(`job={id:'${jobID}',state:'complete',processing_mode:'full',preview_verified:true,video_export_eligible:true,source_sha256:'${hash}',source:{duration_seconds:47.65},video_export:null};config={};$('task').value='pose';$('processingMode').value='full';$('tracking').value='bytetrack';$('model').value='pose';`);
 return {get,run,timers};
}
test('automatic saved-video choice defaults on only for eligible Full Pose ByteTrack and preserves an explicit opt-out',()=>{
 const s=setup();s.run('updateSaveActionVideoOption()');assert.equal(s.get('saveActionVideo').checked,true);assert.equal(s.get('saveActionVideo').disabled,false);
 s.get('saveActionVideo').checked=false;s.run('saveActionVideoChanged();updateSaveActionVideoOption()');assert.equal(s.get('saveActionVideo').checked,false);
 for(const field of ['task','processingMode','tracking']){const previous=s.get(field).value;s.get(field).value=field==='task'?'detect':field==='processingMode'?'replay':'off';s.run('updateSaveActionVideoOption()');assert.equal(s.get('saveActionVideo').disabled,true);assert.equal(s.get('saveActionVideo').checked,false);assert.match(s.get('saveActionVideoNote').textContent,/Full.*Pose.*ByteTrack/);s.get(field).value=previous;}
});
test('auto-save submission includes source-bound Full ROI independently of the replay rule selector and never enables replay auto-save',async()=>{
 const s=setup();s.run(`job=null;$('jobForm').reportValidity=()=>true;$('file').files=[];$('source').value='clip.mp4';$('confidence').value='.05';$('imgsz').value='1280';$('device').value='cpu';$('nms').value='';$('actionRules').value='off';$('saveActionVideo').checked=true;boardROI=[.1,.2,.6,.7];boardROISourceHash='${hash}';boardROIReplayEligible=true;let submitted=[];post=async(path,body)=>{submitted.push(body);return {id:'new',processing_mode:body.processing_mode};};clearViewer=()=>{};renderAnnotations=()=>{};renderArtifacts=()=>{};renderJob=()=>{};poll=()=>{};updateControls=()=>{};`);
 await s.run('startJob({preventDefault(){}})');assert.equal(s.run('submitted[0].save_action_video'),true);assert.equal(s.run('submitted[0].action_rules'),'off');assert.deepEqual(JSON.parse(s.run('JSON.stringify(submitted[0].board_roi)')),[.1,.2,.6,.7]);assert.equal(s.run('submitted[0].board_roi_source_sha256'),hash);
 s.run(`job=null;$('processingMode').value='replay';$('saveActionVideo').checked=true;`);await s.run('startJob({preventDefault(){}})');assert.equal(s.run('submitted[1].save_action_video'),false);
});
test('manual export defaults source range and validates it before dispatch; eligibility comes from the completed job',async()=>{
 const s=setup();s.run('syncVideoExport()');assert.equal(s.get('exportStart').value,'0');assert.equal(s.get('exportEnd').value,'20');
 s.run(`let requests=[];api=async(path,options)=>{requests.push({path,options});return ${JSON.stringify(operation())};};`);
 for(const [a,b] of [['-1','3'],['2','2'],['0','48'],['NaN','5']]){s.get('exportStart').value=a;s.get('exportEnd').value=b;await s.run('startVideoExport()');assert.equal(s.run('requests.length'),0);}
 s.get('exportStart').value='2.125';s.get('exportEnd').value='8.75';s.run(`boardROI=[.1,.2,.6,.7];boardROISourceHash='${hash}';$('task').value='detect';`);await s.run('startVideoExport()');
 assert.equal(s.run('requests[0].path'),`/api/jobs/${jobID}/video-export`);const body=JSON.parse(s.run('requests[0].options.body'));assert.equal(body.start_s,2.125);assert.equal(body.end_s,8.75);assert.equal(body.action_rules,'basic_pose');assert.deepEqual(body.board_roi,[.1,.2,.6,.7]);assert.equal(body.board_roi_source_sha256,hash);
 s.run(`job.video_export_eligible=false;job.video_export_reason='Cần Full Pose ByteTrack';renderVideoExport();`);assert.equal(s.get('startVideoExport').disabled,true);
});
test('manual export snapshots the same selected hand cue and later form changes cannot rewrite it',async()=>{
 const s=setup();s.get('handCue').value='shoulder_any_arm_v1';s.run(`syncVideoExport();let exportBody;let completeRequest;api=(_path,options)=>{exportBody=JSON.parse(options.body);return new Promise(resolve=>completeRequest=resolve);};`);
 const request=s.run('startVideoExport()');assert.equal(s.run('exportBody.hand_cue'),'shoulder_any_arm_v1');
 s.get('handCue').value='nose_both_arms_v1';s.run(`completeRequest(${JSON.stringify(operation())});`);await request;assert.equal(s.run('exportBody.hand_cue'),'shoulder_any_arm_v1');
});
test('export polling updates only its operation while preserving dirty human drafts, form range and source playback',async()=>{
 const s=setup();s.run(`job.video_export=${JSON.stringify(operation())};events=[{label:'HUMAN'}];coverage=[{label:'COVERAGE'}];dirty=true;video.currentTime=7.25;let reloads=0;loadComplete=()=>reloads++;api=async()=>({id:'${jobID}',state:'complete',video_export:${JSON.stringify(operation('complete'))},events:[{label:'WRONG'}]});syncVideoExport();$('exportStart').value='3.5';`);
 await s.run('pollVideoExport(videoExportSession)');assert.equal(s.run('events[0].label'),'HUMAN');assert.equal(s.run('coverage[0].label'),'COVERAGE');assert.equal(s.run('dirty'),true);assert.equal(s.run('video.currentTime'),7.25);assert.equal(s.run('reloads'),0);assert.equal(s.get('exportStart').value,'3.5');assert.equal(s.run('job.state'),'complete');assert.equal(s.run('job.video_export.state'),'complete');
 assert.equal(s.timers.length,0,'terminal export stops polling');
});
test('complete export links use supplied same-job paths in the same tab and never assign a clip to the source video',()=>{
 const s=setup();s.run(`video.src='original-preview';job.video_export=${JSON.stringify(operation('complete'))};syncVideoExport();`);
 assert.equal(s.get('viewVideoExport').href,operation('complete').video_url);assert.equal(s.get('viewVideoExport').target,'_self');assert.equal(s.get('downloadVideoExport').href,operation('complete').video_url.replace('inline=true','inline=false'));assert.equal(s.get('videoExportReport').href,operation('complete').report_url);assert.equal(s.run('video.src'),'original-preview');
 s.run(`job.video_export={...${JSON.stringify(operation('complete'))},video_url:'https://outside.invalid/video.mp4'};syncVideoExport();`);assert.equal(s.get('viewVideoExport').hidden,true);
});
test('switching jobs aborts export requests and a late response cannot publish another job operation or links',async()=>{
 const s=setup();s.run(`let resolve,signal;api=(_path,options)=>{signal=options.signal;return new Promise(r=>resolve=r);};syncVideoExport();`);const saving=s.run('startVideoExport()');
 s.run(`stopVideoExport();jobEpoch++;job={id:'other',state:'complete',video_export:null};resolve(${JSON.stringify(operation('complete'))});`);await saving;
 assert.equal(s.run('signal.aborted'),true);assert.equal(s.run('job.video_export'),null);assert.equal(s.get('viewVideoExport').hidden,true);
});
test('cancel export targets its sub-operation and leaves the completed parent and human annotations intact',async()=>{
 const s=setup();s.run(`job.video_export=${JSON.stringify(operation())};events=[{label:'HUMAN'}];let paths=[];api=async(path,options)=>{paths.push({path,options});return ${JSON.stringify(operation('cancelled'))};};syncVideoExport();`);await s.run('cancelVideoExport()');
 assert.equal(s.run('paths[0].path'),`/api/jobs/${jobID}/video-export/cancel`);assert.equal(s.run('job.state'),'complete');assert.equal(s.run('job.video_export.state'),'cancelled');assert.equal(s.run('events[0].label'),'HUMAN');assert.equal(s.get('cancelVideoExport').disabled,true);
});
test('replay explains sampled limitations and never queues an export or triggers another model run',async()=>{
 const s=setup();s.run(`job.processing_mode='replay';job.video_export_eligible=false;let calls=0;api=()=>{calls++;throw Error('forbidden');};syncVideoExport();`);await s.run('startVideoExport()');assert.equal(s.run('calls'),0);assert.match(s.get('videoExportStatus').textContent,/Replay.*lấy mẫu|lấy mẫu.*Replay/);assert.equal(s.get('startVideoExport').disabled,true);
});
test('one bounded serial status request backs off on failure and a busy export disables a new heavy submission',async()=>{
 const s=setup();s.run(`job.video_export=${JSON.stringify(operation())};api=async()=>{throw Error('network');};syncVideoExport();`);await s.run('pollVideoExport(videoExportSession)');assert.equal(s.timers.length,1);assert.ok(s.timers[0].delay>=1000);s.run('updateControls()');assert.equal(s.get('start').disabled,true);assert.equal(s.get('reviewFields').disabled,true,'existing reviewReady=false is preserved');
 const timeout=s.timers.pop();s.run(`let pendingSignal;api=(_path,options)=>{pendingSignal=options.signal;return new Promise(()=>{});};`);const waiting=s.run('pollVideoExport(videoExportSession)');const deadline=s.timers.find(t=>t.delay>=4000);assert.ok(deadline);deadline.fn();await waiting;assert.equal(s.run('pendingSignal.aborted'),true);assert.ok(s.timers.filter(t=>t.delay<4000).length<=1);
});
test('new job initializes the short source range when completion metadata arrives and existing auto-queued export does not POST twice',()=>{
 const s=setup();s.run(`job.state='processing';job.source=null;job.video_export_eligible=false;syncVideoExport();`);assert.equal(s.get('exportStart').disabled,true);
 s.run(`job.state='complete';job.source={duration_seconds:8.125};job.video_export_eligible=true;job.video_export=${JSON.stringify(operation('queued'))};let calls=0;api=()=>{calls++;};renderJob();`);
 assert.equal(s.get('exportEnd').value,'8.125');assert.equal(s.run('calls'),0);assert.equal(s.timers.length,1);assert.equal(s.get('startVideoExport').disabled,true);
});
test('late pre-cancel status cannot replace cancellation and ROI changes cannot rewrite submitted export options',async()=>{
 const s=setup();s.run(`job.video_export=${JSON.stringify(operation())};syncVideoExport();let oldResolve,pollSignal;api=(path,options)=>{if(path.endsWith('/cancel'))return Promise.resolve(${JSON.stringify(operation('cancelled'))});pollSignal=options.signal;return new Promise(resolve=>oldResolve=resolve);};`);
 const polling=s.run('pollVideoExport(videoExportSession)');await s.run('cancelVideoExport()');s.run(`oldResolve({id:'${jobID}',video_export:${JSON.stringify(operation('complete'))}});`);await polling;assert.equal(s.run('pollSignal.aborted'),true);assert.equal(s.run('job.video_export.state'),'cancelled');assert.equal(s.get('viewVideoExport').hidden,true);
 s.run(`let finish,body;api=(_path,options)=>{body=JSON.parse(options.body);return new Promise(resolve=>finish=resolve);};boardROI=[.1,.2,.6,.7];boardROISourceHash='${hash}';`);const saving=s.run('startVideoExport()');s.run(`boardROI=[.2,.3,.8,.9];finish({...${JSON.stringify(operation())},options:body});`);await saving;
 assert.deepEqual(JSON.parse(s.run('JSON.stringify(job.video_export.options.board_roi)')),[.1,.2,.6,.7]);
});
test('blank end selects source end and a busy409 explains failure without a repeated submission loop',async()=>{
 const s=setup();s.run(`syncVideoExport();$('exportEnd').value='';let calls=0,body;api=async(_path,options)=>{calls++;body=JSON.parse(options.body);const e=Error('Đang có công việc khác');e.status=409;throw e;};`);await s.run('startVideoExport()');assert.equal(s.run('body.end_s'),null);assert.equal(s.run('calls'),1);assert.equal(s.timers.length,0);assert.match(s.get('videoExportStatus').textContent,/công việc khác/);
});
test('export timeout uses export wording and returning from same-tab playback restores links without reloading review',async()=>{
 const s=setup();s.run(`syncVideoExport();api=()=>new Promise(()=>{});`);const saving=s.run('startVideoExport()');s.timers.find(t=>t.delay===5000).fn();await saving;assert.match(s.get('videoExportStatus').textContent,/thời gian/);assert.doesNotMatch(s.get('videoExportStatus').textContent,/replay/i);
 s.run(`job.video_export=${JSON.stringify(operation('complete'))};syncVideoExport();dirty=true;video.currentTime=6.25;let reloads=0;loadComplete=()=>reloads++;stopVideoExport();resumeVideoExport();`);assert.equal(s.get('viewVideoExport').hidden,false);assert.equal(s.run('dirty'),true);assert.equal(s.run('video.currentTime'),6.25);assert.equal(s.run('reloads'),0);
});
test('fresh completed-job load restores last export range while subsequent refresh preserves the user range and dirty review',()=>{
 const s=setup();s.run(`job.video_export={...${JSON.stringify(operation('complete'))},options:{...${JSON.stringify(operation().options)},start_s:22,end_s:32}};syncVideoExport();`);assert.equal(s.get('exportStart').value,'22');assert.equal(s.get('exportEnd').value,'32');
 s.run(`$('exportStart').value='23.125';$('exportEnd').value='30.75';dirty=true;events=[{label:'HUMAN'}];coverage=[{label:'COVERAGE'}];video.currentTime=25;syncVideoExport();`);
 assert.equal(s.get('exportStart').value,'23.125');assert.equal(s.get('exportEnd').value,'30.75');assert.equal(s.run('dirty'),true);assert.equal(s.run('events[0].label'),'HUMAN');assert.equal(s.run('coverage[0].label'),'COVERAGE');assert.equal(s.run('video.currentTime'),25);
 const loading=setup();loading.run(`job.state='queued';job.source=null;syncVideoExport();job.state='complete';job.source={duration_seconds:47.65};job.video_export={...${JSON.stringify(operation('complete'))},options:{...${JSON.stringify(operation().options)},start_s:22,end_s:32}};syncVideoExport();`);assert.equal(loading.get('exportStart').value,'22');assert.equal(loading.get('exportEnd').value,'32');
});
test('fresh last export with null end restores blank end and no prior export retains the short default',()=>{
 const s=setup();s.run(`job.video_export={...${JSON.stringify(operation('complete'))},options:{...${JSON.stringify(operation().options)},start_s:7.5,end_s:null}};syncVideoExport();`);assert.equal(s.get('exportStart').value,'7.5');assert.equal(s.get('exportEnd').value,'');
 const empty=setup();empty.run('syncVideoExport()');assert.equal(empty.get('exportStart').value,'0');assert.equal(empty.get('exportEnd').value,'20');
});
function newJobForm(s){s.run(`$('jobForm').reportValidity=()=>true;$('file').files=[];$('source').value='clip.mp4';$('confidence').value='.05';$('imgsz').value='1280';$('device').value='cpu';$('nms').value='';$('actionRules').value='off';clearViewer=()=>{};renderAnnotations=()=>{};renderArtifacts=()=>{};renderJob=()=>{};poll=()=>{};updateControls=()=>{};let replacements=[];location={href:'http://127.0.0.1:8090/?job=${jobID}&reviewer=group-a&diagnostics=1&extra=keep#viewer',search:'?job=${jobID}&reviewer=group-a&diagnostics=1&extra=keep'};history={state:{remember:'state'},replaceState(state,title,url){replacements.push({state,title,url});const next=new URL(url,location.href);location.href=next.href;location.search=next.search;}};`);}
test('successful new job replaces the same-tab URL reload target and preserves reviewer diagnostics other parameters and fragment',async()=>{
 const s=setup();newJobForm(s);const next='d'.repeat(32);s.run(`post=async()=>({id:'${next}',state:'queued',processing_mode:'full'});`);await s.run('startJob({preventDefault(){}})');
 const url=new URL(s.run('location.href'));assert.equal(url.searchParams.get('job'),next);assert.equal(url.origin,'http://127.0.0.1:8090');assert.equal(url.searchParams.get('reviewer'),'group-a');assert.equal(url.searchParams.get('diagnostics'),'1');assert.equal(url.searchParams.get('extra'),'keep');assert.equal(url.hash,'#viewer');assert.equal(s.run('replacements.length'),1);assert.equal(s.run('replacements[0].state.remember'),'state');assert.equal(s.run('initialJobID()'),next);assert.equal(s.run('job.id'),next);
});
test('failed and outdated new-job submissions leave URL and selected current job unchanged',async()=>{
 const failed=setup();newJobForm(failed);const before=failed.run('location.href');failed.run(`post=async()=>{throw Error('submit failed');};`);await failed.run('startJob({preventDefault(){}})');assert.equal(failed.run('location.href'),before);assert.equal(failed.run('replacements.length'),0);assert.equal(failed.run('job.id'),jobID);
 const s=setup();newJobForm(s);s.run(`let finish;post=()=>new Promise(resolve=>finish=resolve);`);const submission=s.run('startJob({preventDefault(){}})');s.run(`jobEpoch++;job={id:'${'e'.repeat(32)}',state:'complete'};finish({id:'${'d'.repeat(32)}',state:'queued',processing_mode:'full'});`);await submission;assert.equal(s.run('replacements.length'),0);assert.equal(s.run('location.href'),before);assert.equal(s.run('job.id'),'e'.repeat(32));
});
test('stale upload cannot submit a new job and absent history never breaks successful submission',async()=>{
 const s=setup();newJobForm(s);s.run(`$('file').files=[{name:'clip.mp4',size:1}];let finishUpload,posts=0;api=()=>new Promise(resolve=>finishUpload=resolve);post=async()=>{posts++;return {id:'${'d'.repeat(32)}'};};`);const uploading=s.run('startJob({preventDefault(){}})');s.run(`jobEpoch++;finishUpload({source:'new-upload.mp4'});`);await uploading;assert.equal(s.run('posts'),0);assert.equal(s.run('replacements.length'),0);
 const noHistory=setup();newJobForm(noHistory);noHistory.run(`history=undefined;post=async()=>({id:'${'d'.repeat(32)}',processing_mode:'full'});`);await noHistory.run('startJob({preventDefault(){}})');assert.equal(noHistory.run('job.id'),'d'.repeat(32));
});
