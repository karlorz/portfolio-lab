from src.tasker import service


class FakeStore:
    def __init__(self):
        self.wrote = False

    def write_status_mirrors(self, registry):
        self.wrote = True
        return {"backend": "tasker", "tasks": []}


class FakeService:
    def __init__(self):
        self.registry = object()
        self.store = FakeStore()
        self.scheduler_started = False

    def start_background_scheduler(self):
        self.scheduler_started = True
        return None


class FakeApp:
    def __init__(self):
        self.ran = False

    def run(self, host, port):
        self.ran = True


def test_service_once_mode_writes_status_mirrors_without_starting_server(monkeypatch):
    fake_service = FakeService()
    fake_app = FakeApp()
    monkeypatch.setattr(service, "build_service", lambda: (fake_service, fake_app))
    monkeypatch.setattr(service, "configure_logging", lambda: None)

    exit_code = service.main(["--once"])

    assert exit_code == 0
    assert fake_service.store.wrote is True
    assert fake_service.scheduler_started is False
    assert fake_app.ran is False
