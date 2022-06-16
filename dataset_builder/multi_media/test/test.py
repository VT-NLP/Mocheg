import os
from PIL import Image

def test1():
    out_dir="out/running/run8/images"
    imgs=next(os.walk(out_dir))[2]
    for img in imgs:
        try:
            image=Image.open(os.getcwd()+"/"+out_dir+"/"+img)
            width, height = image.size
            print(f'{img} size:{width} {height}')
        except:
            print()

def test2():
    import pandas as pd
  
    # Creating new dataframe
    initial_data = {'First_name': ['Ram', 'Mohan', 'Tina', 'Jeetu', 'Meera'], 
                    'Last_name': ['Kumar', 'Sharma', 'Ali', 'Gandhi', 'Kumari'], 
                    'Marks': [12, 52, 36, 85, 23] }
    
    df = pd.DataFrame(initial_data, columns = ['First_name', 'Last_name', 'Marks'])
    
    # Generate result using pandas
    result = []
    for value in df["Marks"]:
        if value >= 33:
            result.append("Pass")
        elif value < 0 and value > 100:
            result.append("Invalid")
        else:
            result.append("Fail")
        
    df["Result"] = result   
    print(df)

def test3():
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                    'B': 'one one two three two two one three'.split(),
                    'C': np.arange(8), 'D': np.arange(8) * 2})
    print(df)
    #      A      B  C   D
    # 0  foo    one  0   0
    # 1  bar    one  1   2
    # 2  foo    two  2   4
    # 3  bar  three  3   6
    # 4  foo    two  4   8
    # 5  bar    two  5  10
    # 6  foo    one  6  12
    # 7  foo  three  7  14
    found=df.loc[df['A'] == 'bar']
    print(found)
    print(len(found))
    print(found.head(1)["B"])
    

test3()