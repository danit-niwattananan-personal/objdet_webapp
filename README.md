# objdet_webapp

This is a web app that allows user to upload video and it will do inference with YOLOv8 model. The performance is real time capable as the total process time for one image is <60 ms.

To use it, run:
```
streamlit run app.py
```
upload the video (limited to only 200 MB by Streamlit) and hit `Submit`.

To exit the web app after usage, just hit `Ctrl + C` in terminal.