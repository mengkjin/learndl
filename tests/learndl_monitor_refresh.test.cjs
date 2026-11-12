/* Exercise real console JavaScript with two browser-like sessions. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('src/api/learndl_monitor/index.html','utf8').split('<script>')[1].split('</script>')[0];
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const ok=services=>({ok:true,status:200,json:async()=>({services})});
function page(fetch,bus=[]){
 const listeners={},elements={},events={};
 class Channel{
  constructor(){bus.push(this);}
  postMessage(data){for(const other of bus)if(other!==this)queueMicrotask(()=>other.onmessage?.({data}));}
 }
 const context=vm.createContext({fetch,AbortController,queueMicrotask,setTimeout,clearTimeout,setInterval:()=>{},BroadcastChannel:Channel,
  DOMParser:class{parseFromString(){return {querySelector:()=>({content:'new-token'})};}},
  sessionStorage:{setItem(){}},localStorage:{getItem:()=>null,setItem(){}},history:{replaceState(){}},location:{pathname:'/'},
  document:{hidden:false,querySelector:()=>({content:'old-token'}),getElementById:id=>elements[id]??=( {} ),addEventListener:(name,callback)=>events[name]=callback},
  window:{addEventListener:(name,callback)=>listeners[name]=callback},
 });
 vm.runInContext(source,context);
 vm.runInContext('render=()=>{}',context);
 return {context,listeners,events,elements,refresh:()=>vm.runInContext('refresh()',context)};
}
test('a change in one console refreshes the other without navigation',async()=>{
 const bus=[];let state='running',reads=0;
 const fetch=async()=>{reads++;return ok([{id:'tasks',state}]);};
 const first=page(fetch,bus),second=page(fetch,bus);
 await tick();await tick();
 state='stopped';await first.refresh();await tick();await tick();
 assert.equal(vm.runInContext('services[0].state',second.context),'stopped');
 const stableReads=reads;await tick();assert.equal(reads,stableReads,'broadcasts must settle');
});
test('focus and visibility refresh immediately and failures clear on recovery',async()=>{
 let failing=true,calls=0;
 const current=page(async()=>{calls++;if(failing)throw Error('offline');return ok([]);});
 await tick();assert.equal(current.elements.notice.textContent,'offline');
 failing=false;current.listeners.focus();await tick();
 assert.equal(current.elements.notice.textContent,'');
 const before=calls;current.events.visibilitychange();await tick();assert.ok(calls>before);
});
test('overlapping refreshes are serialized and rerun after pending response',async()=>{
 let release,calls=0;
 const current=page(()=>{calls++;return calls===1?new Promise(resolve=>release=resolve):Promise.resolve(ok([]));});
 current.listeners.focus();current.listeners.online();assert.equal(calls,1);
 release(ok([]));await tick();await tick();assert.equal(calls,2);
});
test('an old tab recovers credentials after a server restart',async()=>{
 const calls=[];
 const current=page(async(path,options)=>{
  calls.push(path);
  if(path==='/')return {ok:true,text:async()=>'<meta name="console-token" content="new-token">'};
  return options.headers.Authorization==='Bearer new-token'?ok([]):{status:403,ok:false};
 });
 await tick();await tick();
 assert.deepEqual(calls,['/api/services','/','/api/services']);
 assert.equal(current.elements.notice.textContent,'');
});
