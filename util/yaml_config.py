import yaml


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    def __init__(self, caminho_arquivo):
        self.config = self._load(caminho_arquivo)

    def _load(self, caminho_arquivo):
        with open(caminho_arquivo, 'r') as f:
            return yaml.safe_load(f)

    def get_value(self, section, chave, default_value=None):
        return self.config.get(section, {}).get(chave, default_value)

    def get_section(self, section):
        return self.config.get(section, {})

    def __str__(self):
        return yaml.dump(self.config, default_flow_style=False)


def main():
    config = Config('assets/config.yaml')

    name = config.get_value('general', 'name')
    version = config.get_value('general', 'versaio')
    host = config.get_value('database', 'host')
    port = config.get_value('database', 'port')
    user = config.get_value('database', 'user')
    password = config.get_value('database', 'password')

    anthropic_api_key = config.get_value('anthropic', 'api_key')
    google_api_key = config.get_value('gemini', 'api_key')
    openai_api_key = config.get_value('openai', 'api_key')

    print(f"Name: {name}, Version: {version}")
    print(f"Host: {host}, Port: {port}, User: {user}, Pass: {password}")

    print(f"LLM Provider: Anthropic, API_KEY: {anthropic_api_key}")
    print(f"LLM Provider: Goggle, API_KEY: {google_api_key}")
    print(f"LLM Provider: OpenAi, API_KEY: {openai_api_key}")


if __name__ == "__main__":
    main()
