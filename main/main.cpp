#include <stdio.h>
/* Edge Impulse Arduino examples
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK 0

/* Includes ---------------------------------------------------------------- */
#include "Sound_test_inferencing.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "driver/i2s.h"

#include "servoControl.h"
#include "esp_err.h"

#define EI_WEAK_FN __attribute__((weak))

// I2S Microphone PIN ESP S3

#define I2S_SD 42
#define I2S_SCK 41
#define I2S_WS 40

// I2S Microphone PIN ESP32

// #define I2S_WS 25
// #define I2S_SCK 26
// #define I2S_SD 33

// Servo pins definitions
#define PIN_SERVO_0 GPIO_NUM_4
#define PIN_SERVO_1 GPIO_NUM_16
#define PIN_SERVO_2 GPIO_NUM_18
#define PIN_SERVO_3 GPIO_NUM_19
#define PIN_SERVO_4 GPIO_NUM_21

servoControl myServo0;
servoControl myServo1;
servoControl myServo2;
servoControl myServo3;
servoControl myServo4;

// Servo positions
int positionServo0 = 0;
int positionServo1 = 0;
int positionServo2 = 0;
int positionServo3 = 0;
int positionServo4 = 0;

/** Audio buffers, pointers and selectors */
typedef struct
{
  int16_t *buffer;
  uint8_t buf_ready;
  uint32_t buf_count;
  uint32_t n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 2048;
static signed short sampleBuffer[sample_buffer_size];
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static bool record_status = true;

// prototipes
static void audio_inference_callback(uint32_t n_bytes);
static void capture_samples(void *arg);
static bool microphone_inference_start(uint32_t n_samples);
static bool microphone_inference_record();
static void microphone_inference_end();
static int i2s_init(uint32_t sampling_rate);
static int i2s_deinit();
EI_WEAK_FN EI_IMPULSE_ERROR ei_run_impulse_check_canceled();
uint64_t ei_read_timer_ms();
uint64_t ei_read_timer_us();
void ei_putchar(char c);
__attribute__((weak)) void ei_printf(const char *format, ...);
__attribute__((weak)) void ei_printf_float(float f);
__attribute__((weak)) void *ei_malloc(size_t size);
__attribute__((weak)) void *ei_calloc(size_t nitems, size_t size);
__attribute__((weak)) void ei_free(void *ptr);
void configureServos();
// void ejecuteMovementAction(String palabra);

static void audio_inference_callback(uint32_t n_bytes)
{
  for (int i = 0; i < n_bytes >> 1; i++)
  {
    inference.buffer[inference.buf_count++] = sampleBuffer[i];

    if (inference.buf_count >= inference.n_samples)
    {
      inference.buf_count = 0;
      inference.buf_ready = 1;
    }
  }
}

static void capture_samples(void *arg)
{

  const int32_t i2s_bytes_to_read = (uint32_t)arg;
  size_t bytes_read = i2s_bytes_to_read;

  while (record_status)
  {

    /* read data at once from i2s */
    i2s_read((i2s_port_t)1, (void *)sampleBuffer, i2s_bytes_to_read, &bytes_read, 100);

    if (bytes_read <= 0)
    {
      printf("Error in I2S read : %d", bytes_read);
    }
    else
    {
      if (bytes_read < i2s_bytes_to_read)
      {
        printf("Partial I2S read");
      }

      // scale the data (otherwise the sound is too quiet)
      for (int x = 0; x < i2s_bytes_to_read / 2; x++)
      {
        sampleBuffer[x] = (int16_t)(sampleBuffer[x]) * 8;
      }

      if (record_status)
      {
        audio_inference_callback(i2s_bytes_to_read);
      }
      else
      {
        break;
      }
    }
  }
  vTaskDelete(NULL);
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
  inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

  if (inference.buffer == NULL)
  {
    return false;
  }

  inference.buf_count = 0;
  inference.n_samples = n_samples;
  inference.buf_ready = 0;

  if (i2s_init(EI_CLASSIFIER_FREQUENCY))
  {
    printf("Failed to start I2S!");
  }

  vTaskDelay(pdMS_TO_TICKS(100));

  record_status = true;

  xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void *)sample_buffer_size, 10, NULL);

  return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record()
{
  bool ret = true;

  while (inference.buf_ready == 0)
  {
    vTaskDelay(pdMS_TO_TICKS(10));
  }

  inference.buf_ready = 0;
  return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
  numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);

  return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end()
{
  i2s_deinit();
  free(inference.buffer);
}

static int i2s_init(uint32_t sampling_rate)
{
  // Start listening for audio: MONO @ 8/16KHz
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX),
      .sample_rate = sampling_rate,
      .bits_per_sample = (i2s_bits_per_sample_t)16,
      .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = 0,
      .dma_buf_count = 8,
      .dma_buf_len = 512,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = -1,
  };
  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = -1,
      .data_in_num = I2S_SD};
  esp_err_t ret = 0;

  ret = i2s_driver_install((i2s_port_t)1, &i2s_config, 0, NULL);
  if (ret != ESP_OK)
  {
    printf("Error in i2s_driver_install");
  }

  ret = i2s_set_pin((i2s_port_t)1, &pin_config);
  if (ret != ESP_OK)
  {
    printf("Error in i2s_set_pin");
  }

  ret = i2s_zero_dma_buffer((i2s_port_t)1);
  if (ret != ESP_OK)
  {
    printf("Error in initializing dma buffer with 0");
  }

  return int(ret);
}

static int i2s_deinit()
{
  i2s_driver_uninstall((i2s_port_t)1); // stop & destroy i2s driver
  return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif

extern "C" void app_main(void)
{

  printf("Edge Impulse Inferencing Demo");

  // summary of inferencing settings (from model_metadata.h)
  printf("Inferencing settings:\n");
  printf("\tInterval: ");
  printf("\t %f\n", EI_CLASSIFIER_INTERVAL_MS);
  printf(" ms.\n");
  printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
  printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

  configureServos();

  printf("\nStarting continious inference in 8 seconds...\n");
  vTaskDelay(pdMS_TO_TICKS(2000));

  if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false)
  {
    printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
  }

  printf("Recording...\n");

  while (1)
  {
    bool m = microphone_inference_record();
    if (!m)
    {
      printf("ERR: Failed to record audio...\n");
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK)
    {
      printf("ERR: Failed to run classifier (%d)\n", r);
    }

    // print the predictions
    printf("Predictions ");
    printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
           result.timing.dsp, result.timing.classification, result.timing.anomaly);
    size_t winner = 0;
    float valueTemp = 0;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
    {
      printf("    %s: ", result.classification[ix].label);
      printf("\t %f\n", result.classification[ix].value);
      printf("\n");
      if (valueTemp < result.classification[ix].value)
      {
        valueTemp = result.classification[ix].value;
        winner = ix;
      }
    }
    printf("winner:    %s: \n", result.classification[winner].label);
    printf("\t %f\n", result.classification[winner].value);
    printf("comparacion:   %d \n", (int)result.classification[winner].value > 0.7);
    if (result.classification[winner].value > 0.5)
    {
      // ejecuteMovementAction(result.classification[winner].label);
    }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    printf("    anomaly score: ");
    printf_float(result.anomaly);
    printf("\n");
#endif
  }
}

EI_WEAK_FN EI_IMPULSE_ERROR ei_run_impulse_check_canceled()
{
  return EI_IMPULSE_OK;
}

uint64_t ei_read_timer_ms()
{
  return esp_timer_get_time() / 1000;
}

uint64_t ei_read_timer_us()
{
  return esp_timer_get_time();
}

void ei_putchar(char c)
{
  /* Send char to serial output */
  putchar(c);
}

/**
 *  Printf function uses vsnprintf and output using USB Serial
 */
__attribute__((weak)) void ei_printf(const char *format, ...)
{
  static char print_buf[1024] = {0};

  va_list args;
  va_start(args, format);
  int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
  va_end(args);

  if (r > 0)
  {
    printf(print_buf);
  }
}

__attribute__((weak)) void ei_printf_float(float f)
{
  ei_printf("%f", f);
}

__attribute__((weak)) void *ei_malloc(size_t size)
{
  return malloc(size);
}

__attribute__((weak)) void *ei_calloc(size_t nitems, size_t size)
{
  return calloc(nitems, size);
}

__attribute__((weak)) void ei_free(void *ptr)
{
  free(ptr);
}

void configureServos()
{
  myServo0.attach(PIN_SERVO_0, 400, 2600, LEDC_CHANNEL_0, LEDC_TIMER_0);
  myServo1.attach(PIN_SERVO_1, 400, 2600, LEDC_CHANNEL_1, LEDC_TIMER_0);
  myServo2.attach(PIN_SERVO_2, 400, 2600, LEDC_CHANNEL_2, LEDC_TIMER_0);
  myServo3.attach(PIN_SERVO_3, 400, 2600, LEDC_CHANNEL_3, LEDC_TIMER_0);
  myServo4.attach(PIN_SERVO_4, 400, 2600, LEDC_CHANNEL_4, LEDC_TIMER_0);

  myServo0.write(90);
  myServo1.write(90);
  myServo2.write(90);
  myServo3.write(90);
  myServo4.write(90);
  for (int i = 0; i < 180; i++)
  {
    myServo0.write(i);
    myServo1.write(i);
    myServo2.write(i);
    myServo3.write(i);
    myServo4.write(i);
    vTaskDelay(pdMS_TO_TICKS(20));
  }
  for (int i = 180; i > 0; i--)
  {
    myServo0.write(i);
    myServo1.write(i);
    myServo2.write(i);
    myServo3.write(i);
    myServo4.write(i);
    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

/*void ejecuteMovementAction(String palabra)
{

  if (strcmp(palabra, "left") == 0)
  {
    positionServo0 += 30;
    myServo0.write(positionServo0);
  }

  if (strcmp(palabra, "right") == 0)
  {
    positionServo1 += 30;
    myServo1.write(positionServo1);
  }

  if (strcmp(palabra, "go") == 0)
  {
    positionServo2 += 30;
    myServo2.write(positionServo2);
  }

  if (strcmp(palabra, "no") == 0)
  {
    positionServo3 += 30;
    myServo3.write(positionServo3);
  }

  if (strcmp(palabra, "stop") == 0)
  {
    positionServo4 += 30;
    myServo4.write(positionServo4);
  }

  if (strcmp(palabra, "up") == 0)
  {
    positionServo4 += 30;
    myServo4.write(positionServo4);
  }

  if (strcmp(palabra, "yes") == 0)
  {
    positionServo4 += 30;
    myServo4.write(positionServo4);
  }
}
*/