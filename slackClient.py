#!/usr/bin/env python

from slackclient import SlackClient
import chatbot_softmax as model
import time

# starterbot's ID as an environment variable
BOT_ID = 'U85B52LRH'

# constants
AT_BOT = "<@" + BOT_ID + ">"
EXAMPLE_COMMAND = "do"
test = 0
# instantiate Slack & Twilio clients
slack_client = SlackClient('xoxb-277379088867-S7hFmlvw3rYtjzs7SlHaVn7x')



def handle_command(command, channel):
    """
        Receives commands directed at the bot and determines if they
        are valid commands. If so, then acts on the commands. If not,
        returns back what it needs for clarification.
    """
   
    response = model.get_respone(command,weights,biases)
    '''if not response:
        response = "Not sure what you mean"'''

    
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)


def parse_slack_output(slack_rtm_output):
    """
        The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
    """
    output_list = slack_rtm_output
    print(output_list)	
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, whitespace removed
                return output['text'].split(AT_BOT)[1].strip().lower(), \
                       output['channel']
    return None, None

#weights = []
#biases = []
if __name__ == "__main__":
    
    
    #print(model.get_response("Hi siri"))
   
   
    global weights
    global biases
    
    weights,biases = model.train_model()
    '''command = "What time is the lecture"
    response = model.get_respone(command,weights,biases)
    print(response)'''
    
    READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
    
    if slack_client.rtm_connect():
        print("StarterBot connected and running!")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                handle_command(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")
