    if length.value=="Short":
                max_tokens.value=64
            elif length.value=="Medium":
                max_tokens.value=256
            else:
                max_tokens.value=512

            if creativity.value=="None":
                temperature.value=0
            elif creativity.value=="Medium":
                temperature.value=1
            else:
                temperature.value=2
