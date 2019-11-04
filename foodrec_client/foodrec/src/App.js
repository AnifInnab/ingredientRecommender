import React, { useState } from "react";
import "./App.css";
import IngredientRecommender from "./components/IngredientRecommender";
function App() {
  const [ingredients, setIngredients] = useState([
    "Apple",
    "Pear",
    "Lemon",
    "Mango"
  ]);

  const [inputValue, setInputValue] = useState("");

  return (
    <div className="App">
      <header className="App-header">
        <p>Foodrec</p>
        <input onChange={event => setInputValue(event.target.value)}></input>
        <button
          onClick={() => {
            const newIngredients = [...ingredients];
            newIngredients.splice(0, 0, inputValue);
            setIngredients(newIngredients);
          }}
        >
          add ingredient
        </button>

        <IngredientRecommender
          ingr={ingredients}
          setIngredients={setIngredients}
        ></IngredientRecommender>
      </header>
    </div>
  );
}

export default App;
