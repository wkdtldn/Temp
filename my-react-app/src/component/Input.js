import React, { useState } from "react";

const nickName = {
  fontWeight: 600,
};

function Input() {
  let userName = "사람1";
  const [Message, setMessage] = useState("");

  const onChange = (e) => {
    setMessage(e.target.value);
    console.log(e.target);
  };

  return (
    <div>
      <input onChange={onChange} value={Message} />
      <div>
        <div style={nickName}>{userName}</div>
        <span>{Message}</span>
      </div>
    </div>
  );
}
export default Input;
