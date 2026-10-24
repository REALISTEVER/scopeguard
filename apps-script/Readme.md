# ScopeGuard

ScopeGuard compares a Project Scope with a New Request and classifies the request as IN SCOPE, OUT OF SCOPE, or NEEDS CLARIFICATION.

## Purpose
ScopeGuard helps students, developers, and project teams review proposed changes, identify unclear requirements, and reduce scope creep.

## Model backend
The application uses the supplied pretrained-model backend to interpret project scope and new-request text. The Colab model server and Cloudflare tunnel must be running for model requests to work.

## Run instructions
1. Start the supplied Colab model server and tunnel.
2. Open the deployed Google Apps Script Web App.
3. Confirm that the interface reports `Backend connected`.
4. Enter input containing both `Project Scope:` and `New Request:`.
5. Click `Check Scope`.
6. Review and edit the generated assessment before approving or rejecting it.
7. Only approved assessments are added to `Reviewed Scope Decisions`.

## Deployed application
Deploy your own Apps Script web app and use its URL.

## Demo video
https://youtu.be/USZgNA8DUH0

## Submission security
The submitted `Code.gs` contains `REDACTED_BEFORE_SUBMISSION` instead of the active Colab API secret.
