/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "command_responder.h"

#include "Arduino.h"


void set_leds(String set_to) {
  int state;
  if (set_to == "off"){
    state = HIGH;
  }
  else if (set_to == "on"){
    state = LOW;
  }
  digitalWrite(LEDR, state);
  digitalWrite(LEDG, state);
  digitalWrite(LEDB, state);
}

void have_party() {
  set_leds("off");
  int leds[3] = {LEDR, LEDG, LEDB};
  int first_idx = random(3);
  // Turn on a random, colored LED
  digitalWrite(leds[first_idx], LOW);
  
  int second_idx = random(3);
  // With a prob. of 1/3 turn on a second LED to generate more colors
  if (second_idx != first_idx){
    digitalWrite(leds[second_idx], LOW);
  }
  
  delay(100);
}

// Toggles the built-in LED every inference, and lights a colored LED depending
// on which word was detected.
void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  static bool is_initialized = false;
  if (!is_initialized) {
    pinMode(LED_BUILTIN, OUTPUT);
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    // Ensure the LED is off by default.
    // Note: The RGB LEDs on the Arduino Nano 33 BLE
    // Sense are on when the pin is LOW, off when HIGH.
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    is_initialized = true;
  }
  static int32_t last_command_time = 0;
  static int count = 0;
  static bool is_party = false;
  static bool is_light = false;

  TF_LITE_REPORT_ERROR(error_reporter, "Heard %s (%d) @%dms", found_command,
                         score, current_time);
  
  if (is_new_command) {
    /*TF_LITE_REPORT_ERROR(error_reporter, "Heard %s (%d) @%dms", found_command,
                         score, current_time);*/
    // Switch the `is_light`, `is_party` statements according to the detected keyword.
    if (found_command[0] == 'l') {
      last_command_time = current_time;
      is_light = true;
      is_party = false;
    }

    if (found_command[0] == 'p') {
      last_command_time = current_time;
      is_party = true;
      is_light = false;
    }

    if (found_command[0] == 'a') {
      last_command_time = current_time;
      is_party = false;
      is_light = false;
    }
  }

  // If `party` was detected switch random LED on everytime inference is performed.
  if (is_party) {
    have_party();
  }
  // If `light` was detected switch on all LEDs (generating white light).
  else if (is_light){
    set_leds("on");
  }

  // If last_command_time is non-zero but was >3 seconds ago, zero it
  // and switch off the LED.
  if (last_command_time != 0) {
    if (last_command_time < (current_time - 3000)) {
      last_command_time = 0;
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDG, HIGH);
      digitalWrite(LEDB, HIGH);
    }
    // If it is non-zero but <3 seconds ago, do nothing.
    return;
  }

  // Otherwise, toggle the LED every time an inference is performed.
  ++count;
  if (count & 1) {
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(LED_BUILTIN, LOW);
  }
}

#endif  // ARDUINO_EXCLUDE_CODE
