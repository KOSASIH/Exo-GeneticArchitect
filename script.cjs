// Gene data
const genesData = [
  { name: "Gene 1", function: "Function 1", trait: "Trait 1" },
  { name: "Gene 2", function: "Function 2", trait: "Trait 2" },
  // Add more gene data as needed
];

// Get parameter elements
const geneExpressionSlider = document.getElementById("geneExpression");
const mutationRateSlider = document.getElementById("mutationRate");
const geneInteractionsSelect = document.getElementById("geneInteractions");

// Add event listeners to parameter elements
geneExpressionSlider.addEventListener("input", updateBlueprint);
mutationRateSlider.addEventListener("input", updateBlueprint);
geneInteractionsSelect.addEventListener("change", updateBlueprint);

// Function to update the genetic blueprint based on parameters
function updateBlueprint() {
  const geneExpression = geneExpressionSlider.value;
  const mutationRate = mutationRateSlider.value;
  const geneInteractions = geneInteractionsSelect.value;

  // Clear existing gene elements
  const genesContainer = document.getElementById("genes");
  genesContainer.innerHTML = "";

  // Generate gene elements based on parameters
  for (const geneData of genesData) {
    const geneElement = document.createElement("div");
    geneElement.classList.add("gene");
    geneElement.innerHTML = `
      <h3>${geneData.name}</h3>
      <p>Function: ${geneData.function}</p>
      <p>Trait: ${geneData.trait}</p>
      <p>Gene Expression: ${geneExpression}</p>
      <p>Mutation Rate: ${mutationRate}</p>
      <p>Gene Interactions: ${geneInteractions}</p>
    `;
    genesContainer.appendChild(geneElement);
  }
}

// Initial blueprint update
updateBlueprint();
