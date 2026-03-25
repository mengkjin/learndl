from dataclasses import dataclass , field
from typing import Literal

@dataclass
class Proxy:
    """A basic proxy class"""
    protocal: str
    host: str
    port: int
    
    source: str = 'unknown'
    verified: list[str] = field(default_factory=list)

    def __post_init__(self):
        assert self.protocal in ["http", "https", "socks4", "socks5"] , f"Invalid protocal: {self.protocal}"

    @property
    def addr(self) -> str:
        prot = "http" if self.protocal == "https" else self.protocal
        return f"{prot}://{self.host}:{self.port}"

    def __repr__(self) -> str:
        return f"Proxy({self.addr})"

    def __eq__(self, other: 'Proxy') -> bool:
        return self.addr == other.addr

    def __hash__(self) -> int:
        return hash(self.addr)

    @classmethod
    def from_addr(cls, addr: str , source: str = 'unknown') -> 'Proxy':
        protocal, host, port = addr.split("://")
        assert protocal in ["http", "https", "socks4", "socks5"] , f"Invalid protocal: {protocal}"
        return cls(protocal, host, int(port), source)