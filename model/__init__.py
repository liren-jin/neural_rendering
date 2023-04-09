from .model1.network import Network as Network1
from .model1.rm_renderer import RMRenderer as Renderer1

from .model2.network import Network as Network2
from .model2.rm_renderer import RMRenderer as Renderer2

from .model3.network import Network as Network3
from .model3.rm_renderer import RMRenderer as Renderer3

from .model4.network import Network as Network4
from .model4.rm_renderer import RMRenderer as Renderer4

from .model5.network import Network as Network5
from .model5.rm_renderer import RMRenderer as Renderer5


def get_network(cfg):
    network_type = cfg["network_type"]
    print(f"loading {network_type}\n")
    if network_type == "network1":
        return Network1(cfg)
    elif network_type == "network2":
        return Network2(cfg)
    elif network_type == "network3":
        return Network3(cfg)
    elif network_type == "network4":
        return Network4(cfg)
    elif network_type == "network5":
        return Network5(cfg)


def get_renderer(cfg):
    renderer_type = cfg["renderer_type"]
    print(f"loading {renderer_type}\n")
    if renderer_type == "renderer1":
        return Renderer1.init_from_cfg(cfg)
    elif renderer_type == "renderer2":
        return Renderer2.init_from_cfg(cfg)
    elif renderer_type == "renderer3":
        return Renderer3.init_from_cfg(cfg)
    elif renderer_type == "renderer4":
        return Renderer4.init_from_cfg(cfg)
    elif renderer_type == "renderer5":
        return Renderer5.init_from_cfg(cfg)
