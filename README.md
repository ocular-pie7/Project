1. create-next-app@latest
2. pip install flask  <!-- Flask server to run backend -->
3. pip install flask-cors <!--  to handle CORS -->
4. npm install socket.io-client <!--  Real time communication to back end -->
5. pip install flask-socketio eventlet 
6. npm install lucide-react <!--icons-->
7. pip install psutil
8. npm install leaflet
9. pip install opencv-python
10. pip install pyttsx3

<!--To install all: -->
py -m pip install -r requirements.txt
npm install






Install Dependencies:

pip install --upgrade numpy
pip install --upgrade pandas
pip install --upgrade matplotlib keras opencv-python scikit-learn
pip install tensorflow

pip install pyttsx3 - Text to speech engine

--upgrade  -> For the latest version
Or run: pip install --upgrade pip


NB: After installing : (If dependencies don't automatically reflect in project).
1. Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P).
2. Search for "Python: Select Interpreter."
3. Choose the interpreter where you installed the dependencies (Usually the recommended one).

   Long path error installing any Dependency: 
1. Open PowerShell as Administrator:
Press Win + X and select Windows PowerShell (Admin).
2. Run the following command:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
3. Restart your computer to apply the changes.
4. Retry dependency installation.


<!--To run in powershell: (based on installed python version) -- Run from the root folder Ie: \TrafficSignRecognitionSystem  -->
1. python scripts/TrafficSign_Main.py OR py scripts/TrafficSign_Main.py
2. python scripts/TrafficSign_Test.py  OR scripts/

Front-end server:
1. npm run dev

Back-end flask server:
1. python server/server.py


