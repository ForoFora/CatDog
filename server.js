// получаем модуль Express
const express = require("express");
// создаем приложение
const app = express();

var request = require('request');

app.use(express.static(__dirname + "/public"));

const bodyParser = require('body-parser');

app.use(bodyParser.json({
  limit: '10mb',
  type: 'application/json'
}));

// устанавливаем обработчик для маршрута "/"
app.get("/", function (req, res) {
  res.sendFile(__dirname + '/public/index.html');
});

app.post("/upload_image", function (req, res) {
  if (!req.body) return res.sendStatus(400);

  request.post({
    headers: {
      'content-type': 'application/json'
    },
    url: 'http://127.0.0.1:5000/api/v1.0/img',
    body: JSON.stringify({
      img: req.body.userImg
    })
  }, function (error, response, body) {
    console.log(body);
    let ans;
    try {
      ans = JSON.parse(body);
      console.log(ans[0][0]);
      res.json(ans[0][0]);
    } catch {
      res.json("Sorry. Img Bad :/");
    }
  });
});

// начинаем прослушивание подключений на 3000 порту
console.log("Server:\x1b[32m Start >> http://localhost:3000/ \x1b[0m ");
app.listen(3000);