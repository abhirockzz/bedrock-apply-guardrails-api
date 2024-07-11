import boto3

bedrockRuntimeClient = boto3.client('bedrock-runtime', region_name="us-east-1")

guardrail_id = 'ENTER_GUARDRAIL_ID'
guardrail_version = 'ENTER_GUARDRAIL_VERSION'

input = "I have mild fever. Can Tylenol help?"

def main():
    response = bedrockRuntimeClient.apply_guardrail(guardrailIdentifier=guardrail_id,guardrailVersion=guardrail_version, source='INPUT', content=[{"text": {"text": input}}])

    guardrailResult = response["action"]
    print(f'Guradrail action: {guardrailResult}')

    output = response["outputs"][0]["text"]
    print(f'Final response: {output}')

if __name__ == "__main__":
    main()

