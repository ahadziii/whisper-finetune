import os
from dotenv import load_dotenv

class Config:
    def __init__(self, data = {}):
        self._data = data

    def get(self, name) -> str:
        """Get value from configuration file.

        Example:

            from src import config

            // Returns the value of GOOGLE_CLOUD_PROJECT from .env file
            return config.get('project')

        """
        if name not in self._data:
            raise RuntimeError('Invalid configuration key ' + name)

        return self._data[name]

# Load .env file to memory
load_dotenv()

# Fetch values from the .env file
config = Config({
    # 'project': os.getenv('GOOGLE_PROJECT', 'spokentuntikirjanpito'),
    # 'account': os.getenv('GOOGLE_ACCOUNT', './spokentuntikirjanpito.json'),
    # 'bucket': os.getenv('GOOGLE_BUCKET', 'spokentuntikirjanpito.appspot.com'),
    # 'location': os.getenv('GOOGLE_LOCATION', 'europe-west1'),
    # 'hf_token': os.getenv('HF_TOKEN')
})
