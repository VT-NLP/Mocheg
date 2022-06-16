import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from tqdm.autonotebook import tqdm
from retrieval.utils.config import config
torch.set_num_threads(4)
from transformers import CLIPTokenizer

class ImageSearcher:
    def __init__(self,image_encoder_checkpoint)  :
        #First, we load the respective CLIP model
        self.model = SentenceTransformer(image_encoder_checkpoint)
        # self.model._first_module().max_seq_length =77
        self.tokenizer_for_truncation=  CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        

    def truncate_text(self,text):
        tokens=self.model.tokenize([text])
        decoded_text=self.tokenizer_for_truncation.decode(tokens.input_ids[0][:75],skip_special_tokens =True)
        # decoded_text=decoded_text.replace("<|startoftext|>","")
        # decoded_text=decoded_text.replace("<|endoftext|>","")
        # sequence = self.tokenizer_for_truncation.encode_plus(text, add_special_tokens=False,   
        #                                        max_length=77, 
        #                                        truncation=True, 
        #                                        return_tensors='pt' )
        # return self.tokenizer_for_truncation.decode(sequence.input_ids.detach().cpu().numpy().tolist()[0])
        return decoded_text 
        
        

    def encode_corpus(self,img_folder,img_paths,emb_folder,use_precomputed_embeddings):
        # Now, we need to compute the embeddings
        # To speed things up, we destribute pre-computed embeddings
        # Otherwise you can also encode the images yourself.
        # To encode an image, you can use the following code:
        # from PIL import Image
        # img_emb = model.encode(Image.open(filepath))

        
        emb_filename = 'supplementary/img_corpus_emb.pkl'
        emb_dir=os.path.join(emb_folder,emb_filename)
        if use_precomputed_embeddings: 
            # emb_folder="retrieval/input/embeddings"  
            
            
          
                
            with open(emb_dir, 'rb') as fIn:
                emb_file = pickle.load(fIn)  
                self.img_emb,self.img_names=emb_file["img_emb"],emb_file["img_names"]#,emb_file["img_folder"]
                self.img_folder=img_folder
            print("Images:", len(self.img_names))
        else:
            if img_paths!=None:
                self.img_names=img_paths
            img_len=len(self.img_names)
            print("Images:", len(self.img_names))
            # img_list=
            batch_size=128
            live_num_in_current_batch=0
            live_num=0
            current_image_batch=[]
            total_img_emb= torch.tensor([],device= torch.device('cuda'))
            for filepath in self.img_names:
                image=Image.open(os.path.join(img_folder,filepath))
                current_image_batch.append(image)
                live_num_in_current_batch+=1
                
                if live_num_in_current_batch%batch_size==0:
                    img_emb = self.model.encode(current_image_batch, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
                    total_img_emb=torch.cat([total_img_emb,img_emb],0)
                    live_num_in_current_batch=0
                    current_image_batch=[]
                    live_num+=batch_size
                    print(live_num/img_len)
            self.img_emb = total_img_emb
            # self.img_emb = self.model.encode([Image.open(os.path.join(img_folder,filepath)) for filepath in self.img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
            emb_file = { "img_emb": self.img_emb, "img_names": self.img_names ,"img_folder":img_folder}            
            pickle.dump( emb_file, open(emb_dir , "wb" ) )
            self.img_folder=img_folder
 
    def search(self,query, top_k=3):
        query=self.truncate_text(query)
        
        # First, we encode the query (which can either be an image or a text string)
        query_emb = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False )
        
        # Then, we use the util.semantic_search function, which computes the cosine-similarity
        # between the query embedding and all image embeddings.
        # It then returns the top_k highest ranked images, which we output
        hits = util.semantic_search(query_emb, self.img_emb, top_k=top_k)[0]
        
        if config.verbose==True:
            print(f"Query:{query}")
            for hit in hits:
                # print(self.img_names[hit['corpus_id']])
                image_path=self.img_names[hit['corpus_id']]
                # image = Image.open(image_path)
                # image.show()
                print(image_path)
        return hits
                




