const express = require("express");
const app = express();
const cors = require("cors");
app.use(cors());

app.use(express.json());

let runPy = new Promise(function (success, nosuccess) {
  const { spawn } = require("child_process");
  const pyprog = spawn("python", ["./dataset.py"]);

  pyprog.stdout.on("data", function (data) {
    success(data);
    console.log(data);
  });

  pyprog.stderr.on("data", (data) => {
    nosuccess(data);
  });
});

app.get("/", (req, res) => {
  runPy.then(function (fromRunpy) {
    console.log(fromRunpy);
    res.send(fromRunpy);
  });
  // .catch((err) => {
  //   console.log(err);
  // });
});

const Port = 8082;
app.listen(Port, () => {
  console.log("Port -", Port);
});
