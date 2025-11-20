# Varitas Extension

### Running Locally

To build and load the chrome extension:

1. Go into the extension Directory (TL69---Capstone\extension\wxt-dev-wxt) and run "npm run build".
   This will generate the build files in the .output directory
2. Open chrome and go to the manage extensions page "chrome://extensions/" and turn on developer mode
3. Click load unpacked and find the .output folder from above and select the chrome-mv3 folder
4. Future changes to the extension can simply be run by rebuilding the extension and reloading

### Known Issues

- highlighting/underlining is inconsistent and may not work
  - chance of failing increases with longer passages of text + text that is broken up (new lines, embedded links)
  - big issue since "scan entire page" functionality will break highlighting/underlining a lot
- cannot highlight with underlines active
  - connects to first point, underlining breaks up text -> can no longer highlight
  - smaller issue since having both highlight + underline is repeated info
