const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const html=fs.readFileSync(require('node:path').join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
function setup(search,stored){const context=vm.createContext({URLSearchParams,location:{search}});vm.runInContext(`let stored=${JSON.stringify(stored)},message='',remembered=[];function storageGet(){return stored;}function storageSet(k,v){remembered.push(v);}function storageRemove(){stored=null;}function error(m){message=m;}
${html.split(/\r?\n/).find(l=>l.startsWith('function initialJobID('))}`,context);return s=>vm.runInContext(s,context);}
test('shared job link wins over browser history and remembers the selected canonical job',()=>{const run=setup('?job='+'A'.repeat(32),'b'.repeat(32));assert.equal(run('initialJobID()'),'a'.repeat(32));assert.equal(run('remembered[0]'),'a'.repeat(32));assert.equal(run('message'),'');});
test('invalid or ambiguous job link does not fall back to opening another stored job',()=>{for(const value of ['?job=..%2Fsecret','?job=','?job='+'a'.repeat(32)+'&job='+'b'.repeat(32)]){const run=setup(value,'b'.repeat(32));assert.equal(run('initialJobID()'),null);assert.match(run('message'),/liên kết|32/);assert.equal(run('remembered.length'),0);}});
test('ordinary entry preserves the existing valid browser history fallback',()=>{const run=setup('','c'.repeat(32));assert.equal(run('initialJobID()'),'c'.repeat(32));});
