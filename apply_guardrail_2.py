from sagemaker.predictor import retrieve_default
import boto3

bedrockRuntimeClient = boto3.client('bedrock-runtime', region_name="us-east-1")

guardrail_id = 'ENTER_GUARDRAIL_ID'
guardrail_version = 'ENTER_GUARDRAIL_VERSION'
endpoint_name = 'ENTER_SAGEMAKER_ENDPOINT'

messages = [
  { "role": "system","content": "When requested for a doctor appointment, reply with a confirmation of the appointment along with a random appointment ID and a random patient health insurance ID. Don't ask additional questions."}
]

# messages = [
#   { "role": "system","content": "When requested for a doctor appointment, reply with a confirmation of the appointment along with a random appointment ID. Don't ask additional questions"}
# ]

def main():
    
    #prompt = "Can you help me with medicine suggestions for mild fever?"
    prompt = "I need an appointment with Dr. Smith for 4 PM tomorrow."

    safe, output = safeguard_check(prompt,'INPUT')

    if safe == False:
        print("Final response:", output)
        return


    messages.append({"role": "user", "content": prompt})
    inputText = build_prompt(messages)

    predictor = retrieve_default(endpoint_name)

    payload = {
        "inputs": inputText,
        "parameters": {
            "max_new_tokens": 256,
            "top_p": 0.9,
            "temperature": 0.6
        }
    }

    print("Invoking Sagemaker endpoint")

    response = predictor.predict(payload)
    sm_output = response[0]["generated_text"]

    #print(f'Sagemaker response:\n{sm_output}')

    response_text = extract_response(sm_output)

    safe, output = safeguard_check(response_text,'OUTPUT')

    if safe == False:
        print("Final response:\n", output)
        return
    else:
        print("Final response:\n", response_text)


def safeguard_check(input, source):

    print(f'Checking {source} - {input}')
    
    response = bedrockRuntimeClient.apply_guardrail(guardrailIdentifier=guardrail_id,guardrailVersion=guardrail_version, source=source, content=[{"text": {"text": input}}])

    #print(f'Guardrail check response:\n {response}')
    
    guardrailResult = response["action"]

    if guardrailResult == "NONE":
        print("Result: No Guardrail intervention")
        return True, ""
    elif guardrailResult == "GUARDRAIL_INTERVENED":
        reason = response["assessments"]
        print("Guardrail intervention due to:", reason)
        #print(response)
        output = response["outputs"][0]["text"]
        return False, output


def build_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")
 
    return startPrompt + "".join(conversation) + endPrompt

def extract_response(text):
    parts = text.split("[/INST]")
    response = parts[1].strip()
    #print(f'Extracted text from Sagemaker response:\n{response}')
    return response

if __name__ == "__main__":
    main()

