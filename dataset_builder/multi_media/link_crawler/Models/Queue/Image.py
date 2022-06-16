from link_crawler.Models.Model import Model


class Image(Model):
   

    def __init__(self,run_dir):
        Model.__init__(self)
        self.file_path=run_dir+"/queue_gopostie.txt"
        self.fetch()


