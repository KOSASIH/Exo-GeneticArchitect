import openai

def generate_genetic_blueprint(num_genes, gene_functions, desired_traits):
    prompt = f"On a distant world, we are architecting the genetic blueprint of life.\n\nNumber of genes: {num_genes}\n\nGene functions: {gene_functions}\n\nDesired traits: {desired_traits}\n\n"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    blueprint = response.choices[0].text.strip()
    
    return blueprint

# User input for specific parameters
num_genes = int(input("Enter the desired number of genes: "))
gene_functions = input("Enter the gene functions (comma-separated): ").split(",")
desired_traits = input("Enter the desired traits (comma-separated): ").split(",")

# Generate the genetic blueprint
genetic_blueprint = generate_genetic_blueprint(num_genes, gene_functions, desired_traits)

# Format the genetic blueprint as markdown code
markdown_code = f"```\n{genetic_blueprint}\n```"

print(markdown_code)
