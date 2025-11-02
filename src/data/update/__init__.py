from .basic import BasicUpdater

class AffiliatedDataUpdater:
    @classmethod
    def update(cls):
        BasicUpdater.import_updaters()
        for name , updater in BasicUpdater.registry.items():
            updater.update()

    @classmethod
    def update_rollback(cls , rollback_date : int):
        BasicUpdater.import_updaters()
        for name , updater in BasicUpdater.registry.items():
            updater.update_rollback(rollback_date)