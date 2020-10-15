# -*- coding: utf-8 -*-


#!pip install gTTS

from gtts import gTTS

#chooses the language, English(‘en’)

convert = gTTS(text='I like this NLP book', lang='en', slow=False) 
  
# Saving the converted audio in a mp3 file named 
myobj.save("audio.mp3")