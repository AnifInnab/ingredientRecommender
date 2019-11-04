import React from "react";

function IngredientRecommender(props) {
  return (
    <div>
      {props.ingr.map((ingredient, i) => {
        return (
          <div
            key={i}
            onClick={() => {
              const newIngredients = [...props.ingr];
              newIngredients.splice(i, 1);
              props.setIngredients(newIngredients);
            }}
          >
            {ingredient + " "}
          </div>
        );
      })}
    </div>
  );
}

export default IngredientRecommender;
