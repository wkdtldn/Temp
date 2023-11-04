import React, { useState } from "react";

function Input() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const onChange = (e) => {
    setUsername(e.target.value);
    setPassword(e.target.value);
    console.log(e.target);
  };

  const deleteText = () => {
    setUsername("");
    setPassword("");
  };

  return (
    <div>
      <input onChange={onChange} value={username} />
      <input onChange={onChange} value={password} />
      <p>{username}</p>
      <p>{password}</p>
      <button onClick={deleteText}>X</button>
    </div>
  );
}
export default Input;
