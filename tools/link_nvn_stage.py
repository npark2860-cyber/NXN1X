"""Link a named foundation stage from current objects; no emulator execution."""
from pathlib import Path
import importlib.util, json, hashlib, shutil, sys
ROOT=Path(__file__).resolve().parents[1]
stage=sys.argv[1]; assert stage in ('f3','f4','f4r','f5','f5r','f5s','f5t','f5u','f5v','f5w')
HERE=ROOT/f'_xfer/nvn_foundation_{stage}_20261117'; HERE.mkdir(exist_ok=True)
if stage in ('f5r','f5s','f5t','f5u','f5v','f5w'):
 gate=json.loads((HERE/'static-gate.json').read_text()); assert gate['status']=='PASS'
 validated=json.loads((HERE/'validated-source-hashes.json').read_text())
 assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in validated.items()), 'Validated inputs changed'
 for name in ('tests.json','guard-tests.json','extracted-tests.json','shader-stop-tests.json','release-object-tests.json'):
  assert all(r['exit_code']==0 for r in json.loads((HERE/name).read_text())), name
 if stage=='f5s':
  protocol=json.loads((HERE/'protocol-tests.json').read_text())
  assert len(protocol)==18 and all(r['exit_code']==0 for r in protocol), 'F5S normal/failure integration gate'
 if stage=='f5t':
  protocol=json.loads((HERE/'protocol-tests.json').read_text())
  compositor=json.loads((HERE/'compositor-tests.json').read_text())
  assert len(protocol)==27 and all(r['exit_code']==0 for r in protocol), 'F5T actual pool caller gate'
  assert len(compositor)==10 and all(r['exit_code']==0 for r in compositor), 'F5T compositor/LSFG caller gate'
 if stage in ('f5u','f5v','f5w'):
  protocol=json.loads((HERE/'protocol-tests.json').read_text())
  compositor=json.loads((HERE/'compositor-tests.json').read_text())
  assert len(protocol)==36 and all(r['exit_code']==0 for r in protocol), 'F5U actual image/pool caller gate'
  assert len(compositor)==10 and all(r['exit_code']==0 for r in compositor), 'F5U retained compositor regression gate'
 if stage=='f5v':
  assert json.loads((HERE/'lazy-tests.json').read_text())['status']=='PASS', 'F5V deferred recipe caller gate'
 if stage=='f5w':
  for name,count in [('owner-route-tests.json',23),('owner-selector-tests.json',2)]:
   rows=json.loads((HERE/name).read_text())
   assert len(rows)==count and all(r['exit_code']==0 for r in rows),name
  assert not (ROOT/'source/src/video_core/renderer_vulkan/nvn_pipeline_recipe.h').exists(), 'F5V shader change must stay reverted'
template=(ROOT/'_xfer/nvn_foundation_f2_20261117/link_candidate.py').read_text()
template=template.replace('foundation-f2',f'foundation-{stage}').replace('foundation-f1r.exe','foundation-f2r.exe')
p=HERE/'link_candidate.py'; p.write_text(template)
spec=importlib.util.spec_from_file_location('stage_link',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
paths=set(json.loads((ROOT/'_xfer/nvn_foundation_f2r_20261117/source-hashes.json').read_text()))
paths.update(['host/dm/windows/nvn_image_eligibility.h','host/dm/windows/nvn_memory_observe.h','source/src/video_core/vulkan_common/nvn_memory_admission.h'])
paths.update(['source/src/video_core/texture_cache/texture_cache_base.h','source/src/video_core/texture_cache/texture_cache.h'])
paths.update(['host/dm/windows/nvn_pool_reclaim.h','host/dm/windows/nvn_memory_content.h'])
if stage in ('f5','f5r','f5s','f5t','f5u','f5v','f5w'):
 paths.update(['host/dm/windows/nvn_memory_transfer.h','host/dm/windows/nvn_memory_transfer.c','host/dm/windows/nvn_image_archive.h','source/src/video_core/vulkan_common/nvn_memory_pressure.h','source/src/nvn_host/CMakeLists.txt'])
if stage in ('f5r','f5s','f5t','f5u','f5v','f5w'): paths.update(validated)
hashes={}
for name in paths:
 p=ROOT/name
 if p.is_file():
  hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  out=HERE/'snapshot'/name; out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,out)
(HERE/'source-hashes.json').write_text(json.dumps(hashes,indent=2))
m.main()
bin=ROOT/'build/release/bin'; side=bin/f'NXN1X-NVN-foundation-{stage}.turnip.json'
assert not side.exists(); shutil.copy2(bin/'NXN1X-NVN-foundation-f2r.turnip.json',side)
(HERE/'package.json').write_text(json.dumps(dict(executable=json.loads((HERE/'executable.json').read_text()),sidecar=str(side),sidecar_sha256=hashlib.sha256(side.read_bytes()).hexdigest(),driver_sha256=hashlib.sha256((bin/'turnip/f2r/vulkan_freedreno.dll').read_bytes()).hexdigest()),indent=2))
