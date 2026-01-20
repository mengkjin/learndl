def setup_mosek_lic():
    import shutil
    from src.proj import MACHINE , PATH , Logger
    source_lic = PATH.resource.joinpath('mosek.lic')
    target_lic = MACHINE.mosek_lic_path

    if target_lic and target_lic.exists():
        if source_lic.stat().st_mtime > target_lic.stat().st_mtime:
            shutil.copy(target_lic , target_lic.with_suffix('.bak'))
            shutil.copy(source_lic , target_lic)
            Logger.success(f'{source_lic} is newer, replaced {target_lic} with it')

def setup_solvers():
    setup_mosek_lic()
            