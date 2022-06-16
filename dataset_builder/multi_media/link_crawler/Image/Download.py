import errno
import os.path
import link_crawler.Models.Queue.Image 
import link_crawler.Models.Complete.Image 
from link_crawler.Image import Save
import urllib.request
import re
class Download():
    

    def __init__(self,run_dir,prefix, take=None, links=None, path="images"):
        self.folder=run_dir
        self.queue = link_crawler.Models.Queue.Image.Image(run_dir)
        self.complete =  link_crawler.Models.Complete.Image.Image(run_dir)
        self.limit = take
        self.path = os.path.join(self.folder , path)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if isinstance(links, set):
            self.links = links
        elif  isinstance(links, list):
            self.links = set(links)
        else:
            self.links = self.queue.links
        self.prefix=prefix
    """
        Start Downloading file
        :rtype: object
        """
    def start(self) -> object:
        
        if isinstance(self.limit, int) and len(self.links) >= self.limit:
            links = sorted(self.links)[0:self.limit]
        else:
            links = self.links

        for img_id,file in enumerate(links):
            try:
                saving_name=self.gen_name(file,img_id,self.prefix)
                img = Save.Save(file, saving_name,self.path)
                img.save()
                self.complete.add(file)
            except Exception as e:
                print(f"{e} for {file}")
            
        self.save()
        return self

    """
        Update links txt file
        :return : None
        """
    def save(self) :
        
        self.queue.links = self.queue.links.difference(self.complete.links)
        self.queue.save()
        self.complete.save()
        
    def gen_name(self,file_name,img_id,prefix) :
        
        base_name = os.path.basename(urllib.request.urlparse(file_name).path)
        length=len( base_name)
        threshold=80
        if length>threshold:
            base_name=base_name[length-threshold:length]
        saving_name= prefix+"-"+str(img_id).zfill(2)+"-"+base_name
        
        if not  re.search("\.[a-z][a-z][a-z]",  saving_name) and not  re.search("\.[a-z][a-z][a-z][a-z]", saving_name):
            saving_name+=".jpg"
        return saving_name
        # filename, file_extension =os.path.splitext('/path/to/somefile.ext')