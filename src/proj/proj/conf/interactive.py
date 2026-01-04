from src.__version__ import __version__
class InteractiveAppConfig:
    @property
    def version(self) -> str:
        """version of the app"""
        return __version__

    @property
    def recommended_explorer(self) -> str:
        """recommended explorer of the app"""
        return 'chrome'

    @property
    def page_title(self) -> str:
        """title of the app"""
        return 'Learndl'

    @property
    def pending_features(self) -> list[str]:
        """pending features of the app"""
        return []

    @property
    def auto_refresh_interval(self) -> int:
        """auto refresh interval of the app"""
        return 0

Interactive = InteractiveAppConfig()
