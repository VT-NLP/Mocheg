import os.path
import urllib.parse
import errno


class Model:
    file_path = ''

    def __init__(self):
        self.links = set()

    def fetch(self):
        """
            Load links from file and set to Set()
            :return: object
        """
        self.init_dir()
        with open(self.file_path, 'r') as f:
            for line in f:
                self.links.add(line.replace('\n', ''))
        return self.links

    def save(self):
        """

        :type links: object
        """
        with open(self.file_path, 'w') as f:
            for line in sorted(self.links):
                f.write(line + "\n")
        return True

    def has(self, link):
        if not link in self.links:
            return False
        else:
            return True

    def add(self, link):
        self.links.add(link)
        return self

    def remove(self, link):
        self.links.discard(link)
        return self

    def init_dir(self):
        dir_name = os.path.dirname(self.file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.isfile(self.file_path):
            with open(self.file_path, 'w') as f:
                f.write('')
