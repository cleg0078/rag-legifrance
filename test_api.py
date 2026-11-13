import requests

# Quick authentication test
r = requests.post(
    "https://sandbox-oauth.piste.gouv.fr/api/oauth/token",
    data={
        "grant_type"   : "client_credentials",
        "client_id"    : "73d0cdfb-7f25-4115-b081-75e31a9e4e31",
        "client_secret": "458a5da3-5bea-4f58-a086-961be2f6ff4a",
        "scope"        : "openid",
    }
)
print(r.json())
# → {"access_token": "eyJ...", "expires_in": 3600}
