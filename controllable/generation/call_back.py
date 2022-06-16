from transformers import TrainerCallback  
from datetime import datetime 

def todayAt(hr, min=0, sec=0, micros=0):
    now = datetime.now()
    return now.replace(hour=hr, minute=min, second=sec, microsecond=micros)    
 
 
   
class TimerCallback(TrainerCallback):
    def on_step_begin(self, args , state , control , **kwargs):
       
        
        if args.end_before_wake_up:
        
            timeNow =  datetime.now()
            if timeNow > todayAt(7) and timeNow < todayAt(23):
                control.should_training_stop =True 
                control.should_epoch_stop =True 
                control.should_evaluate =False 
   
   
        # threshold = datetime.strptime(threshold_time_str, '%b %d %Y %I:%M%p')
        # if current_date>threshold  :
            
    