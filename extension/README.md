# Varitas Extension

### Installation

1. Go to `chrome://extensions/` on your browser, enable "Developer mode" and click "Load unpacked"
2. Select the `VeritasExtension/` folder located in `TL69---Capstone/extension/`
3. Reload the page/restart the browser

Note: Both models (bias detection, claim verification) should be located at `http://localhost:8000`.  
User auth should be located at `http://localhost:8080`.  
See respective folders for starting up servers.

### Running Locally (development)

To build and load the chrome extension:

1. Go into the extension Directory (TL69---Capstone\extension\wxt-dev-wxt) and run "npm run build".
   This will generate the build files in the .output directory
2. Open chrome and go to the manage extensions page "chrome://extensions/" and turn on developer mode
3. Click load unpacked and find the .output folder from above and select the chrome-mv3-dev folder
4. Run "npm run dev" and reload the page/restart the browser
