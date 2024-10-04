from datetime import datetime
import os


class storageBucket:
    def __init__(self, parent='outputs'):
        self.parent = parent
        self.bucket_name = self.generate_bucket_name()
        self.create_main_folder()

    def generate_bucket_name(self):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return timestamp

    def create_main_folder(self):
        os.mkdir(f'{self.parent}/{self.bucket_name}')

    def create_folder(self, name):
        os.mkdir(f'{self.parent}/{self.bucket_name}/{name}')

        def get_path():
            return f'{self.parent}/{self.bucket_name}/{name}'

        def write_file(data, _name, type='bytes'):
            if type == 'bytes':
                f = open(f'{self.parent}/{self.bucket_name}/{name}/{_name}', 'wb')
                f.write(data)
            else:
                f = open(f'{self.parent}/{self.bucket_name}/{name}/{_name}', 'a')
                f.write(data)

        return write_file, get_path
