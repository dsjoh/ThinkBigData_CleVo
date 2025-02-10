# ThinkBigData_CleVo

## Project

2024 광운대학교 산학연계SW프로젝트 씽크빅데이터팀 </br>
(CleVo : AI를 활용한 영어 발음 학습 서비스) </br> </br>
광운대학교 정보융합학부 </br>
오승준 김민서 신원석 설수빈 이원빈

### PDF : [LINK](https://github.com/dsjoh/ThinkBigData_CleVo/blob/dev/%5BSW%EC%82%B0%ED%95%99%EC%97%B0%EA%B3%84%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%5D%EC%94%BD%ED%81%AC%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%ED%8C%80_%EC%B5%9C%EC%A2%85%EB%B3%B4%EA%B3%A0%EC%84%9C-%EB%B3%B5%EC%82%AC.pdf)
</br>

## License

해당 프로젝트는 OpenAI와 ETRI에서 제공하는 API의 결과물을 포함하고 있습니다. </br>
해당 프로젝트는 SpringBoot, H2, Lombok, Apache Commons Text, CMUdict(Carnegie Mellon University Pronouncing Dictionary) 등의 오픈소스 SW를 활용하여 제작되었습니다. </br></br>
This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details. </br>


## Poster

![CleVo](https://github.com/user-attachments/assets/8a27aa59-56ff-4d02-ab64-f7538842dde0)


## How To Run

※ 본 프로젝트는 보안상의 이유로 API Key와 계정 정보가 포함된 env.yml, application-dev.yml, application-test.yml 파일이 포함되지 않았습니다. </br> </br>
1. GitHub에서 해당 프로젝트 파일을 받는다 </br>
2. YML 파일들을 따로 받아서 추가해준다 </br>
3. Docker설치 후 Docker 터미널에서 </br>
docker pull redis </br>
docker run --name my-redis -p 6379:6379 -d redis </br>
으로 redis 서버 설치 후 실행해준다 </br>
4. springboot 백엔드서버를 실행한다 </br>
5. http://localhost:8080/MAIN/main.html 접속 </br>


2. springboot 백엔드서버 실행해준다 </br>
3. http://localhost:8080/MAIN/main.html 접속
