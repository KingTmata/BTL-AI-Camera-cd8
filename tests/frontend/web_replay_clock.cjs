const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict'),{test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
function setup(){
 let now=1000;const nodes=new Map(),replies=[];
 const get=id=>{if(!nodes.has(id))nodes.set(id,{dataset:{},style:{},textContent:'',getContext(){return {clearRect(){}};}});return nodes.get(id);};
 const context=vm.createContext({document:{getElementById:get},performance:{now:()=>now},Headers,AbortController,URLSearchParams,TextDecoder,Uint8Array,Float64Array,DataView,setTimeout(){return 1;},clearTimeout(){},fetch:async()=>{const reply=replies.shift();if(reply instanceof Error)throw reply;now+=reply.rtt;return {ok:true,json:async()=>({clock_id:reply.id||'server',monotonic_seconds:reply.stamp})};}});
 vm.runInContext(html.split('<script>')[1].split('if(videoFrameAPI){')[0],context);
 const run=code=>vm.runInContext(code,context);
 return {run,get,reply(...items){replies.push(...items);},time(value){now=value;}};
}
test('three exchanges choose minimum RTT and bound a known server/client offset',async()=>{
 const s=setup();s.run(`let session={id:'one',epoch:0,controller:new AbortController(),timing:createReplayTiming('one'),clockID:null,calibration:null,startClickMs:500};`);
 s.reply({rtt:20,stamp:101.01},{rtt:4,stamp:101.022},{rtt:10,stamp:101.029});
 await s.run('calibrateReplayClock(session)');
 assert.equal(s.run('session.calibration.rtt_ms'),4);
 assert.ok(s.run('session.calibration.offset_lower_seconds<=100&&session.calibration.offset_upper_seconds>=100'));
 s.time(1200);s.run(`recordReplayCommit(session,{sequence:0,source_frame_index:0,source_pts:0,time_base:'1/1000',timestamp_seconds:0,source_release_monotonic:101.1,publish_monotonic:101.15},1200);`);
 const record=JSON.parse(s.run('JSON.stringify(session.timing.records[0])'));
 assert.ok(record.source_to_commit_ms.lower<=100&&record.source_to_commit_ms.upper>=100);
 assert.ok(record.publish_to_commit_ms.lower<=50&&record.publish_to_commit_ms.upper>=50);
 assert.equal(s.run('session.timing.start_click_to_first_commit_ms'),700);
 assert.equal(s.run('session.timing.browser_paint_latency_ms'),null);
});
test('missing clock and negative/invalid timing remain unknown without changing the source sample',async()=>{
 const s=setup();s.run(`let session={id:'one',epoch:0,controller:new AbortController(),timing:createReplayTiming('one'),clockID:null,calibration:null,startClickMs:100};let row={sequence:0,source_frame_index:6,source_pts:200,time_base:'1/1000',timestamp_seconds:.2,source_release_monotonic:-1,publish_monotonic:NaN};let before=JSON.stringify(row);`);
 s.reply(Error('missing'),Error('missing'),Error('missing'));await s.run('calibrateReplayClock(session)');
 s.run('recordReplayCommit(session,row,1200)');
 assert.equal(s.run('JSON.stringify(row)===before'),true);
 assert.equal(s.run('session.timing.records[0].source_to_commit_ms'),null);
 assert.equal(s.run('session.timing.lifetime.source_to_commit.valid_count'),0);
 assert.equal(s.run('session.timing.start_click_to_first_commit_ms'),null);
});
test('raw records cap does not truncate lifetime counts or invent lifetime percentiles',()=>{
 const s=setup();s.run(`let session={id:'one',epoch:0,controller:new AbortController(),timing:createReplayTiming('one'),clockID:'server',calibration:{clock_id:'server',offset_lower_seconds:100,offset_upper_seconds:100.001,rtt_ms:1,received_client_ms:1000},startClickMs:null};for(let i=0;i<10005;i++)recordReplayCommit(session,{sequence:i,source_frame_index:i,source_pts:i,time_base:'1/1000',timestamp_seconds:i,source_release_monotonic:101,publish_monotonic:101.1},5000);`);
 assert.equal(s.run('session.timing.records.length'),10000);assert.equal(s.run('session.timing.records_dropped'),5);
 assert.equal(s.run('session.timing.lifetime.commits'),10005);
 assert.equal(s.run('session.timing.lifetime.source_to_commit.upper_over_3s_count'),10005);
 assert.equal(s.run('session.timing.records_truncated'),true);
 assert.equal(s.run('session.timing.lifetime.p95_ms'),undefined);
});
test('old-job calibration cannot attach and clock changes discard the previous timing epoch',async()=>{
 const s=setup();s.run(`let session={id:'one',epoch:0,controller:new AbortController(),timing:createReplayTiming('one'),clockID:'old',calibration:{clock_id:'old'},startClickMs:null};session.timing.lifetime.commits=7;invalidateReplayClock(session,'new');`);
 assert.equal(s.run('session.calibration'),null);assert.equal(s.run('session.timing.clock_resets'),1);
 assert.equal(s.run('session.timing.discarded_previous_epoch_commits'),7);
 s.reply({rtt:3,stamp:101.001,id:'new'},{rtt:3,stamp:101.004,id:'new'},{rtt:3,stamp:101.007,id:'new'});
 const calibrating=s.run('calibrateReplayClock(session)');s.run('jobEpoch++;session.controller.abort();');await calibrating;
 assert.equal(s.run('session.calibration'),null);
});
test('invalid/future timestamps and expired calibration never become zero latency samples',()=>{
 const s=setup();s.run(`let session={id:'one',epoch:0,controller:new AbortController(),timing:createReplayTiming('one'),clockID:'server',calibration:{clock_id:'server',offset_lower_seconds:100,offset_upper_seconds:100.001,rtt_ms:1,received_client_ms:1000},startClickMs:0};let row={sequence:0,source_frame_index:0,source_pts:0,time_base:'1/1000',timestamp_seconds:0,source_release_monotonic:-1,publish_monotonic:101.1};recordReplayCommit(session,row,1200);recordReplayCommit(session,{...row,sequence:1,source_release_monotonic:102,publish_monotonic:103},1200);recordReplayCommit(session,{...row,sequence:2,source_release_monotonic:101,publish_monotonic:101.1},62000);`);
 assert.equal(s.run('session.timing.lifetime.source_to_commit.valid_count'),0);
 assert.equal(s.run('session.timing.lifetime.source_to_commit.unknown_count'),3);
 assert.equal(s.run('session.timing.lifetime.source_to_commit.max_upper_ms'),null);
 assert.equal(s.run('session.timing.start_click_to_first_commit_ms'),null);
});
