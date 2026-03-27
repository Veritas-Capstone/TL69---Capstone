# Varitas User Authentication

### Running Locally

Follow the `example.env` file to create a `.env` file with `DATABASE=mongodb-string-here`.
To get the string: https://send.bitwarden.com/#3mhWWg925kKDhrQSAWnUIg/gHmxJayVeOOgZI_zMXm_xw
Password is veritas

Run `npx concurrently "uvicorn server.server_combined:app --reload --port 8000" "cd user-auth && npm start"`
from project root. This will start up the user auth backend along with backend for the models.
