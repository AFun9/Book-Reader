package com.afun9.bookreader.app

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.afun9.bookreader.ui.LibraryScreen
import com.afun9.bookreader.ui.ReaderScreen
import com.afun9.bookreader.ui.VoiceStudioScreen

object Routes {
    const val Library = "library"
    const val Reader = "reader"
    const val VoiceStudio = "voice_studio"
}

@Composable
fun BookReaderApp() {
    MaterialTheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background,
        ) {
            val navController = rememberNavController()

            NavHost(
                navController = navController,
                startDestination = Routes.Library,
            ) {
                composable(Routes.Library) {
                    LibraryScreen(
                        onOpenReader = { navController.navigate(Routes.Reader) },
                        onOpenVoiceStudio = { navController.navigate(Routes.VoiceStudio) },
                    )
                }

                composable(Routes.Reader) {
                    ReaderScreen(
                        onBack = { navController.popBackStack() },
                        onOpenVoiceStudio = { navController.navigate(Routes.VoiceStudio) },
                    )
                }

                composable(Routes.VoiceStudio) {
                    VoiceStudioScreen(
                        onBack = { navController.popBackStack() },
                    )
                }
            }
        }
    }
}
