# Команда CPUSHNIKI — Решение Hackathon Go1

## Сборка Docker-образа

    docker build --platform linux/amd64 -t cput1 .

## Проверка протокола ping

    docker run --rm --network none cput1 ping
