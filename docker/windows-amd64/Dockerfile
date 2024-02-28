# See: https://levelup.gitconnected.com/debugging-go-inside-docker-using-visual-studio-code-and-remote-containers-5c3724fe87b9
# See for available variants: https://hub.docker.com/_/golang?tab=tags
ARG VARIANT=1.22.0-windowsservercore
FROM golang:${VARIANT}

WORKDIR c:/workspace

ENTRYPOINT ["powershell", "while ($true) {Start-Sleep -Seconds 60 }"]
