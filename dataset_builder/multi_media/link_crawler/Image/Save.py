import urllib.request
import os.path
import re 
import os
from PIL import Image
import socket
thresholdWidth=512
thresholdHeight=512
timeout = 20
import time
socket.setdefaulttimeout(timeout)
from urllib.parse import quote
class Save:
    def __init__(self, file_name,saving_name, path=''):
        """
        SaveFile Constructor
        :param file_name:
        :param path
        :rtype: object
        """
        self.file_name = file_name
        self.base_name=saving_name
        
        

        self.path = path
        init_request()

    """
        Download from web and save it to local folder
        :rtype: object
        """
    def save(self) -> object:
        
         
        if len(self.path) > 0:
            full_file_path = self.path + '/' + self.base_name
        else:
            full_file_path = self.base_name
        url_name = self.file_name.replace(" ","")
        urllib.request.urlretrieve( url_name , full_file_path) 
        filter_by_size(full_file_path)
         
         
        

def init_request():
    # Adding information about user agent
    opener=urllib.request.build_opener()
    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)

def filter_by_size(full_file_path):
    filename=os.getcwd()+"/"+full_file_path
    try:
        image=Image.open(filename)
    except Exception as e:
       
        wait_for_file(filename)
        try:
            image=Image.open(filename)
        except Exception as e1:
            os.remove(filename)
            raise e1
             
    width, height = image.size
    if (width <= thresholdWidth or
            height <= thresholdHeight):
        os.remove(filename)

 
def is_locked(filepath):
    locked = None
    file_object = None
    if os.path.exists(filepath):
        try:
            buffer_size = 8
            # Opening file in append mode and read the first 8 characters.
            file_object = open(filepath, 'a', buffer_size)
            if file_object:
                locked = False
        except IOError as message:
            locked = True
        finally:
            if file_object:
                file_object.close()
    return locked

def wait_for_file(filepath):
    wait_time = 1
    while is_locked(filepath):
        time.sleep(wait_time)